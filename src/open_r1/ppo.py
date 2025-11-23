# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Callable, List

import datasets
import torch
import transformers
from datasets import load_dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import PPOConfig
from open_r1.rewards import (
    accuracy_reward,
    code_reward,
    format_reward,
    get_code_format_reward,
    get_cosine_scaled_reward,
    get_repetition_penalty_reward,
    len_reward,
    reasoning_steps_reward,
    tag_count_reward,
)
from open_r1.utils import get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import PPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

logger = logging.getLogger(__name__)


class RewardFunctionsWrapper(torch.nn.Module):
    """
    Wrapper that allows PPOTrainer to use multiple reward functions directly.
    This wraps the reward functions to be compatible with TRL's PPOTrainer,
    which expects a model with a .score() method.
    
    Args:
        reward_funcs: List of reward function callables
        reward_weights: Optional list of weights for each reward function (default: equal weights)
        tokenizer: Tokenizer for decoding completions
    """
    
    def __init__(self, reward_funcs: List[Callable], reward_weights: Optional[List[float]] = None, tokenizer=None):
        super().__init__()
        self.reward_funcs = reward_funcs
        self.tokenizer = tokenizer
        self.base_model_prefix = "base_model"
        
        # Create a dummy base_model that mimics a transformer backbone
        # The get_reward() function will call this to get hidden states
        class DummyBackbone(torch.nn.Module):
            def forward(self, input_ids, attention_mask=None, **kwargs):
                # Return a dummy output with hidden_states attribute
                # This is what get_reward() expects to process through our score() method
                batch_size, seq_len = input_ids.shape
                hidden_dim = 768  # Dummy hidden dimension
                
                # Create dummy hidden states (batch_size, seq_len, hidden_dim)
                dummy_hidden = torch.zeros(batch_size, seq_len, hidden_dim, device=input_ids.device)
                
                # Store the input_ids so score() can use them to compute rewards
                class DummyOutput:
                    def __init__(self, hidden_states, input_ids):
                        self.hidden_states = hidden_states
                        self._input_ids = input_ids
                
                return DummyOutput([dummy_hidden], input_ids)
        
        self.base_model = DummyBackbone()
        
        if reward_weights is None:
            reward_weights = [1.0 / len(reward_funcs)] * len(reward_funcs)
        else:
            total = sum(reward_weights)
            reward_weights = [w / total for w in reward_weights]
        self.reward_weights = reward_weights
    
    def forward(self, *args, **kwargs):
        """Dummy forward - PPOTrainer uses .score() instead."""
        return torch.tensor([0.0])
    
    def score(self, hidden_states, **kwargs):
        """
        Compute rewards from the input sequences using reward functions.
        
        Args:
            hidden_states: Tensor of shape (batch_size, seq_len, hidden_dim) 
                          (passed by get_reward() from model output)
            **kwargs: Additional kwargs from PPOTrainer
        
        Returns:
            Tensor of rewards with shape (batch_size, seq_len) or (batch_size, 1)
        """
        # Get batch size from hidden states
        batch_size = hidden_states.shape[0]
        device = hidden_states.device
        
        # We need to get the original input_ids to decode and compute rewards
        # Unfortunately, get_reward() doesn't pass them, so we'll work with what we have
        # For now, return a tensor of zeros with the right shape - the caller expects (batch, seq_len)
        # and will select specific indices based on sequence_length
        seq_len = hidden_states.shape[1] if len(hidden_states.shape) > 1 else 1
        
        # Return dummy reward logits
        # Shape should be (batch_size, seq_len) for compatibility
        reward_logits = torch.zeros(batch_size, seq_len, device=device, dtype=hidden_states.dtype)
        
        # Try to use cached input_ids if available
        if hasattr(self, '_current_input_ids') and self._current_input_ids is not None:
            input_ids = self._current_input_ids
            for i, input_id_seq in enumerate(input_ids):
                if self.tokenizer is not None:
                    text = self.tokenizer.decode(input_id_seq, skip_special_tokens=True)
                else:
                    text = str(input_id_seq)
                
                # Compute combined reward
                total_reward = 0.0
                for reward_func, weight in zip(self.reward_funcs, self.reward_weights):
                    try:
                        reward = reward_func(text)
                        total_reward += weight * float(reward)
                    except Exception as e:
                        logger.debug(f"Error computing reward: {e}")
                
                # Set this reward for the last valid token
                if seq_len > 0:
                    reward_logits[i, -1] = total_reward
        
        return reward_logits
    
    def set_input_ids(self, input_ids):
        """Store input_ids for use in score() method."""
        self._current_input_ids = input_ids


class ValueModelWrapper(torch.nn.Module):
    """
    Wrapper for value model to add .score() method expected by PPOTrainer.
    """
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.base_model_prefix = "model"  # For compatibility with PPOTrainer
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def score(self, logits, **kwargs):
        """
        Compute value scores from hidden states.
        Returns dummy value for now - PPO will override with actual rewards.
        """
        # Return scalar value per batch element
        if isinstance(logits, torch.Tensor):
            return torch.mean(logits, dim=-1)
        return torch.zeros(1, dtype=torch.float32)


@dataclass
class PPOScriptArguments(ScriptArguments):
    """
    Script arguments for the PPO training script.

    This mirrors the GRPO script arguments but adds an optional `value_model_name_or_path`
    argument that PPO can use for a separate value network (optional).
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "tag_count"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', tag_count', 'code', 'code_format'"
        },
    )
    reward_weights: list[float] = field(
        default_factory=lambda: None,
        metadata={"help": "Optional weights for each reward function (will be normalized)"},
    )
    cosine_min_value_wrong: float = field(default=0.0)
    cosine_max_value_wrong: float = field(default=-0.5)
    cosine_min_value_correct: float = field(default=0.5)
    cosine_max_value_correct: float = field(default=1.0)
    cosine_max_len: int = field(default=1000)
    repetition_n_grams: int = field(default=3)
    repetition_max_penalty: float = field(default=-1.0)
    code_language: str = field(default="python")
    value_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional separate value model for PPO (if omitted, trainer uses internal value head)."},
    )
    max_prompt_length: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum prompt length for tokenization (defaults to training_args.max_prompt_length)"},
    )


def main(script_args, training_args, model_args):
    set_seed(training_args.seed)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    tokenizer = get_tokenizer(model_args, training_args)
    
    # Set padding_side to 'left' for generation with Flash Attention
    # This is required for Qwen2 with Flash Attention to avoid issues with batched generation
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    REWARD_FUNCS_REGISTRY = {
        "accuracy": accuracy_reward,
        "format": format_reward,
        "reasoning_steps": reasoning_steps_reward,
        "cosine": get_cosine_scaled_reward(
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        ),
        "repetition_penalty": get_repetition_penalty_reward(
            ngram_size=script_args.repetition_n_grams, max_penalty=script_args.repetition_max_penalty
        ),
        "length": len_reward,
        "code": code_reward,
        "code_format": get_code_format_reward(language=script_args.code_language),
        "tag_count": tag_count_reward,
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    def make_conversation(example):
        prompt = []
        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})
        prompt.append({"role": "user", "content": example["problem"]})
        return {"prompt": prompt}

    dataset = dataset.map(make_conversation)

    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")
    
    # Tokenize prompts for PPO - apply tokenizer to format as chat
    def tokenize_fn(example):
        # Convert prompt (list of dicts) to text using tokenizer's chat template
        # PPOTrainer will add the generation prompt itself, so we DON'T add it here
        text = tokenizer.apply_chat_template(
            example["prompt"],
            tokenize=False,
            add_generation_prompt=False  # PPOTrainer will add generation prompt
        )
        # Tokenize the text
        tokens = tokenizer(
            text,
            max_length=script_args.max_prompt_length or training_args.max_prompt_length,
            truncation=True,
            padding=False  # Don't pad yet - will be done by collator
        )
        return tokens
    
    logger.info("*** Tokenizing dataset ***")
    # Store solutions before removing columns
    solutions_map = {}
    if "solution" in dataset["train"].column_names:
        for idx, example in enumerate(dataset["train"]):
            solutions_map[idx] = example["solution"]
    
    # Remove non-tokenized columns and tokenize
    columns_to_remove = [col for col in dataset["train"].column_names if col not in ["input_ids", "attention_mask", "prompt"]]
    dataset = dataset.map(
        tokenize_fn,
        batched=False,
        remove_columns=["problem", "solution", "answer", "level", "prompt"] if "problem" in dataset["train"].column_names else []
    )
    
    # Log sample from dataset to understand structure
    sample = dataset["train"][0]
    logger.info(f"Sample dataset entry keys: {sample.keys()}")
    logger.info(f"Sample input_ids length: {len(sample['input_ids'])}")
    logger.info(f"Sample input_ids (first 30 tokens): {sample['input_ids'][:30]}")
    decoded_sample = tokenizer.decode(sample['input_ids'], skip_special_tokens=False)
    logger.info(f"Sample decoded prompt: {decoded_sample[:300]}")
    
    # Store solutions as metadata for reward computation
    training_args.solutions_map = solutions_map


    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    training_args.model_init_kwargs = model_kwargs

    # Create reward model wrapper from reward functions
    logger.info(f"Creating reward model wrapper from {len(reward_funcs)} reward functions")
    reward_weights = script_args.reward_weights if script_args.reward_weights else None
    reward_model = RewardFunctionsWrapper(reward_funcs, reward_weights)

    # Load model (PPO requires actual model objects, not model paths)
    logger.info(f"*** Loading model {model_args.model_name_or_path} ***")
    from transformers import AutoModelForCausalLM
    
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype if torch_dtype != torch.bfloat16 else torch.float32,  # Use float32 for loading
    )
    
    # Cast to bfloat16 if needed
    if torch_dtype == torch.bfloat16:
        policy_model = policy_model.bfloat16()
    
    # Create or load value model
    if script_args.value_model_name_or_path:
        logger.info(f"*** Loading value model {script_args.value_model_name_or_path} ***")
        value_model_base = AutoModelForCausalLM.from_pretrained(
            script_args.value_model_name_or_path,
            revision=model_args.model_revision,
            trust_remote_code=model_args.trust_remote_code,
            torch_dtype=torch_dtype if torch_dtype != torch.bfloat16 else torch.float32,
        )
        if torch_dtype == torch.bfloat16:
            value_model_base = value_model_base.bfloat16()
        value_model = ValueModelWrapper(value_model_base)
    else:
        # Use policy model as value model too (common PPO setup)
        logger.info("Using policy model as value model")
        value_model = ValueModelWrapper(policy_model)

    # Workaround for PPOTrainer with DeepSpeed: reset AcceleratorState singleton
    # so PPOTrainer can create its own Accelerator instance properly
    from accelerate.state import AcceleratorState
    # Clear the global state if it exists (happens when running under accelerate launch)
    if AcceleratorState._shared_state:
        AcceleratorState._shared_state.clear()

    trainer = PPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=policy_model,
        ref_model=None,  # PPO will create ref_model internally from model
        reward_model=reward_model,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_train_split],  # Use train dataset for eval to avoid errors
        value_model=value_model,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
    )

    logger.info("*** Train ***")
    
    # Patch the get_reward function to work with our reward functions
    # The standard get_reward() expects a model with .score() that processes hidden states
    # We need to compute rewards from text completions instead
    from trl.trainer.utils import get_reward as original_get_reward
    from trl.trainer.utils import first_true_indices, truncate_response
    
    def patched_get_reward(model, query_responses, pad_token_id, context_length):
        """
        Modified get_reward that handles our reward functions wrapper.
        This properly extracts completions and calls reward functions.
        
        Args:
            model: Reward model
            query_responses: (batch_size, seq_len) tensor of token ids
            pad_token_id: Token ID for padding
            context_length: Length of the prompt (where generation starts)
        """
        # If it's our reward wrapper, handle specially
        if isinstance(model, RewardFunctionsWrapper):
            batch_size = query_responses.shape[0]
            seq_len = query_responses.shape[1]
            device = query_responses.device
            
            # Log some debug info on first batch
            if not hasattr(patched_get_reward, 'logged'):
                logger.info(f"PPO Reward computation - context_length: {context_length}, seq_len: {seq_len}, batch_size: {batch_size}")
                logger.info(f"Sample prompt tokens: {query_responses[0, :context_length].tolist()[:20]}...")
                logger.info(f"Sample completion tokens: {query_responses[0, context_length:min(context_length+20, seq_len)].tolist()}")
                patched_get_reward.logged = True
            
            # Compute sequence lengths
            attention_mask = query_responses != pad_token_id
            # Find where padding starts (first pad token after context)
            sequence_lengths = torch.full((batch_size,), seq_len - 1, device=device, dtype=torch.long)
            for i in range(batch_size):
                # Find first pad token after context_length
                for j in range(context_length, seq_len):
                    if query_responses[i, j] == pad_token_id:
                        sequence_lengths[i] = max(j - 1, context_length)
                        break
            
            # Extract completions (only the generated part after prompt)
            completions = query_responses[:, context_length:]
            completion_texts = tokenizer.batch_decode(completions, skip_special_tokens=True)
            
            if not hasattr(patched_get_reward, 'logged_text'):
                logger.info(f"Sample completion text: {completion_texts[0][:200]}")
                patched_get_reward.logged_text = True
            
            rewards_list = []
            for text in completion_texts:
                total_reward = 0.0
                reward_breakdown = {}
                
                for reward_func, weight in zip(model.reward_funcs, model.reward_weights):
                    try:
                        # Format completion for reward function
                        # Reward functions expect [{"role": "assistant", "content": "..."}]
                        completion_formatted = [{"role": "assistant", "content": text}]
                        reward = reward_func([completion_formatted])
                        # reward is returned as list
                        if isinstance(reward, list):
                            reward = reward[0]
                        reward_value = weight * float(reward)
                        total_reward += reward_value
                        reward_breakdown[reward_func.__name__] = (float(reward), reward_value)
                    except Exception as e:
                        logger.debug(f"Error computing reward: {e}")
                        reward_breakdown[reward_func.__name__] = (0.0, 0.0)
                
                if len(rewards_list) < 2 and total_reward > 0:
                    logger.info(f"Reward breakdown for completion: {reward_breakdown}, total: {total_reward}")
                
                rewards_list.append(total_reward)
            
            # Create reward logits tensor (batch_size, seq_len)
            reward_logits = torch.zeros(batch_size, seq_len, device=device, dtype=torch.float32)
            
            # Set the reward at the last valid position of each sequence
            for i in range(batch_size):
                seq_end = min(sequence_lengths[i].item(), seq_len - 1)
                reward_logits[i, seq_end] = rewards_list[i]
            
            # Return (reward_logits, final_rewards, sequence_lengths)
            final_rewards = torch.tensor(rewards_list, device=device, dtype=torch.float32)
            
            return reward_logits, final_rewards, sequence_lengths
        
        # For other models (value model), use original get_reward
        return original_get_reward(model, query_responses, pad_token_id, context_length)
    
    # Patch get_reward in the trainer module
    import trl.trainer.ppo_trainer as ppo_trainer_module
    ppo_trainer_module.get_reward = patched_get_reward
    
    # Run standard training
    train_result = trainer.train()
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    kwargs = {"dataset_name": script_args.dataset_name, "tags": ["open-r1", "ppo"]}
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((PPOScriptArguments, PPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
