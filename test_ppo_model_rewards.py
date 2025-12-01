#!/usr/bin/env python3
"""Test to validate PPO model is generating correct rewards through the RewardFunctionsWrapper."""

import sys
sys.path.insert(0, '/home/rl-group10/training_scripts/open-rs/src')

import torch
from transformers import AutoTokenizer
from open_r1.rewards import format_reward, tag_count_reward
from open_r1.ppo import RewardFunctionsWrapper

# Test completions
good_completion = """<think>
Let me solve this step by step. We have a square with vertices at (0,0), (0,3), (3,3), and (3,0).
The area of the square is 3 * 3 = 9.
We need to find the probability that x + y < 5.
The region where x + y < 5 is below the line y = 5 - x.
</think>

<answer>
The line x + y = 5 passes through (2, 3) and (3, 2).
The area where x + y < 5 within the square is a triangle and a trapezoid.
Actually, the area is half of the region below the line.
The probability is 7/18.
</answer>"""

bad_completion = """The probability is 7/18 because the square has area 9 and the region where x+y<5 has area 3.5."""

# Partial: has closing tag but missing opening tag
partial_completion = """Let me think about this problem.

</think>

The answer is 7/18."""

# Missing both tags entirely
no_tags_completion = """Let me work through this. The square has area 9. Looking at the region where x+y<5, I need to find what portion satisfies this constraint. The answer is 7/18."""

print("=" * 80)
print("TEST 1: Direct Reward Functions (Baseline)")
print("=" * 80)

# Format as expected by reward functions
good_formatted = [{"role": "assistant", "content": good_completion}]
bad_formatted = [{"role": "assistant", "content": bad_completion}]
partial_formatted = [{"role": "assistant", "content": partial_completion}]
no_tags_formatted = [{"role": "assistant", "content": no_tags_completion}]

# Test reward functions directly
format_reward_good = format_reward([good_formatted])[0]
format_reward_bad = format_reward([bad_formatted])[0]
format_reward_partial = format_reward([partial_formatted])[0]
format_reward_no_tags = format_reward([no_tags_formatted])[0]

tag_count_reward_good = tag_count_reward([good_formatted])[0]
tag_count_reward_bad = tag_count_reward([bad_formatted])[0]
tag_count_reward_partial = tag_count_reward([partial_formatted])[0]
tag_count_reward_no_tags = tag_count_reward([no_tags_formatted])[0]

print(f"Good completion:")
print(f"  format_reward:    {format_reward_good:.3f}")
print(f"  tag_count_reward: {tag_count_reward_good:.3f}")
print(f"  combined:         {format_reward_good + tag_count_reward_good:.3f}")

print(f"\nBad completion:")
print(f"  format_reward:    {format_reward_bad:.3f}")
print(f"  tag_count_reward: {tag_count_reward_bad:.3f}")
print(f"  combined:         {format_reward_bad + tag_count_reward_bad:.3f}")

print(f"\nPartial completion (missing opening tag):")
print(f"  format_reward:    {format_reward_partial:.3f}")
print(f"  tag_count_reward: {tag_count_reward_partial:.3f}")
print(f"  combined:         {format_reward_partial + tag_count_reward_partial:.3f}")

print(f"\nNo tags completion:")
print(f"  format_reward:    {format_reward_no_tags:.3f}")
print(f"  tag_count_reward: {tag_count_reward_no_tags:.3f}")
print(f"  combined:         {format_reward_no_tags + tag_count_reward_no_tags:.3f}")

print("\n" + "=" * 80)
print("TEST 2: RewardFunctionsWrapper (as used in PPO Trainer)")
print("=" * 80)

# Create tokenizer and wrapper
tokenizer = AutoTokenizer.from_pretrained("gpt2")
reward_funcs = [format_reward, tag_count_reward]
reward_weights = [0.5, 0.5]  # Equal weights

wrapper = RewardFunctionsWrapper(reward_funcs, reward_weights, tokenizer)

# Tokenize completions
print("Tokenizing completions...")
good_tokens = tokenizer.encode(good_completion, return_tensors="pt")
bad_tokens = tokenizer.encode(bad_completion, return_tensors="pt")
partial_tokens = tokenizer.encode(partial_completion, return_tensors="pt")
no_tags_tokens = tokenizer.encode(no_tags_completion, return_tensors="pt")

# Test the wrapper's score method with mock hidden states
print("\nTesting score() method with mock hidden states...")

# Create dummy hidden states (batch_size=1, seq_len=variable, hidden_dim=768)
batch_size = 1
hidden_dim = 768

good_hidden = torch.randn(batch_size, good_tokens.shape[1], hidden_dim)
bad_hidden = torch.randn(batch_size, bad_tokens.shape[1], hidden_dim)
partial_hidden = torch.randn(batch_size, partial_tokens.shape[1], hidden_dim)
no_tags_hidden = torch.randn(batch_size, no_tags_tokens.shape[1], hidden_dim)

# Set input_ids in wrapper (this simulates what PPO trainer would do)
wrapper.set_input_ids(good_tokens)
good_reward_wrapper = wrapper.score(good_hidden)
print(f"Good completion reward from wrapper:           {good_reward_wrapper[0, -1].item():.3f}")

wrapper.set_input_ids(bad_tokens)
bad_reward_wrapper = wrapper.score(bad_hidden)
print(f"Bad completion reward from wrapper:            {bad_reward_wrapper[0, -1].item():.3f}")

wrapper.set_input_ids(partial_tokens)
partial_reward_wrapper = wrapper.score(partial_hidden)
print(f"Partial completion (missing opening tag):      {partial_reward_wrapper[0, -1].item():.3f}")

wrapper.set_input_ids(no_tags_tokens)
no_tags_reward_wrapper = wrapper.score(no_tags_hidden)
print(f"No tags completion reward from wrapper:        {no_tags_reward_wrapper[0, -1].item():.3f}")

print("\n" + "=" * 80)
print("TEST 3: Reward Computation Step-by-Step")
print("=" * 80)

# Manually compute what the wrapper should be doing
print("Manual computation of weighted rewards:")

good_format = format_reward([good_formatted])[0]
good_tags = tag_count_reward([good_formatted])[0]
good_manual = 0.5 * good_format + 0.5 * good_tags
print(f"Good: 0.5*{good_format:.3f} + 0.5*{good_tags:.3f} = {good_manual:.3f}")

bad_format = format_reward([bad_formatted])[0]
bad_tags = tag_count_reward([bad_formatted])[0]
bad_manual = 0.5 * bad_format + 0.5 * bad_tags
print(f"Bad: 0.5*{bad_format:.3f} + 0.5*{bad_tags:.3f} = {bad_manual:.3f}")

partial_format = format_reward([partial_formatted])[0]
partial_tags = tag_count_reward([partial_formatted])[0]
partial_manual = 0.5 * partial_format + 0.5 * partial_tags
print(f"Partial: 0.5*{partial_format:.3f} + 0.5*{partial_tags:.3f} = {partial_manual:.3f}")

no_tags_format = format_reward([no_tags_formatted])[0]
no_tags_tags = tag_count_reward([no_tags_formatted])[0]
no_tags_manual = 0.5 * no_tags_format + 0.5 * no_tags_tags
print(f"No tags: 0.5*{no_tags_format:.3f} + 0.5*{no_tags_tags:.3f} = {no_tags_manual:.3f}")

print("\n" + "=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)

# Check consistency
print("\nComparing wrapper output with manual computation:")
good_match = abs(good_reward_wrapper[0, -1].item() - good_manual) < 0.001
bad_match = abs(bad_reward_wrapper[0, -1].item() - bad_manual) < 0.001
partial_match = abs(partial_reward_wrapper[0, -1].item() - partial_manual) < 0.001
no_tags_match = abs(no_tags_reward_wrapper[0, -1].item() - no_tags_manual) < 0.001

print(f"Good completion:          {'✓ MATCH' if good_match else '✗ MISMATCH'}")
print(f"Bad completion:           {'✓ MATCH' if bad_match else '✗ MISMATCH'}")
print(f"Partial completion:       {'✓ MATCH' if partial_match else '✗ MISMATCH'}")
print(f"No tags completion:       {'✓ MATCH' if no_tags_match else '✗ MISMATCH'}")

# Check reward ordering - good should be highest, then partial, then bad/no_tags
print("\nValidating reward ordering (good > partial > bad, no_tags):")
order_correct = good_manual > partial_manual and partial_manual >= bad_manual and partial_manual >= no_tags_manual
print(f"good ({good_manual:.3f}) > partial ({partial_manual:.3f}) >= bad ({bad_manual:.3f}), no_tags ({no_tags_manual:.3f}): {'✓ CORRECT' if order_correct else '✗ INCORRECT'}")

# Check wrapper ordering
wrapper_order_correct = (good_reward_wrapper[0, -1].item() > partial_reward_wrapper[0, -1].item() and 
                         partial_reward_wrapper[0, -1].item() >= bad_reward_wrapper[0, -1].item() and
                         partial_reward_wrapper[0, -1].item() >= no_tags_reward_wrapper[0, -1].item())
print(f"Wrapper ordering matches: {'✓ YES' if wrapper_order_correct else '✗ NO'}")

if all([good_match, bad_match, partial_match, no_tags_match, order_correct, wrapper_order_correct]):
    print("\n✓✓✓ ALL TESTS PASSED - Rewards are computing correctly! ✓✓✓")
else:
    print("\n✗✗✗ SOME TESTS FAILED - Check reward computation ✗✗✗")

print("\n" + "=" * 80)
print("USAGE IN PPO TRAINING")
print("=" * 80)
print("""
The RewardFunctionsWrapper is used by PPOTrainer as follows:

1. During generation: Model generates text
2. The generated text is tokenized (input_ids)
3. wrapper.set_input_ids(input_ids) is called to cache the IDs
4. wrapper.score(hidden_states) is called to compute rewards
5. Rewards are used to update policy via PPO loss

Key points:
- The wrapper accepts multiple reward functions with weights
- Each reward function should take completions in format: [{"role": "assistant", "content": text}]
- Rewards are normalized by weights
- The score() method returns a tensor of shape (batch_size, seq_len)
- PPO trainer uses the last token's reward by default

To debug reward issues:
- Run this test to validate individual reward functions
- Check that rewards align with your training goal
- Monitor reward values during training (they should be in reasonable range)
- Use different weight combinations to emphasize certain rewards
""")
