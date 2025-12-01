#!/usr/bin/env python3
"""
Test script to evaluate the trained PPO model on sample problems.
Loads the model and generates solutions with reasoning.
"""

import sys
sys.path.insert(0, '/home/rl-group10/training_scripts/open-rs/src')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

# Model configuration
MODEL_PATH = "/home/rl-group10/training_scripts/open-rs/data/OpenRS-PPO_v1"

# Test problems
TEST_PROBLEMS = [
    {
        "problem": "A point (x, y) is randomly and uniformly chosen inside the square with vertices (0,0), (0,3), (3,3), and (3,0). What is the probability that x + y < 5?",
        "expected_answer": "7/18"
    },
    {
        "problem": "What is 2 + 2?",
        "expected_answer": "4"
    },
    {
        "problem": "Solve for x: 2x + 3 = 11",
        "expected_answer": "4"
    },
]

# System prompt
SYSTEM_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer, and put your final answer within \\boxed{{}} . The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. Note that respond by English, NOT use other languages."""

def load_model_and_tokenizer():
    """Load the trained model and tokenizer."""
    print("=" * 80)
    print("Loading trained model and tokenizer...")
    print("=" * 80)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
        
        # Set tokenizer padding
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        
        print(f"✓ Model loaded: {MODEL_PATH}")
        print(f"✓ Model dtype: {model.dtype}")
        print(f"✓ Model device: {model.device}")
        print()
        
        return model, tokenizer
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        sys.exit(1)

def generate_solution(model, tokenizer, problem, max_new_tokens=3584):
    """Generate a solution for a given problem."""
    
    # Create chat messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem}
    ]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    generated_text = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    
    return generated_text

def extract_answer(text):
    """Extract the answer from the generated text."""
    # Try to extract from <answer> tags
    if "<answer>" in text and "</answer>" in text:
        start = text.find("<answer>") + len("<answer>")
        end = text.find("</answer>")
        return text[start:end].strip()
    
    # Try to extract from \\boxed{}
    if "\\boxed{" in text:
        start = text.find("\\boxed{") + len("\\boxed{")
        end = text.find("}", start)
        return text[start:end].strip()
    
    # Return last line as fallback
    lines = text.strip().split('\n')
    return lines[-1] if lines else ""

def main():
    """Main testing function."""
    
    # Load model
    model, tokenizer = load_model_and_tokenizer()
    
    print("=" * 80)
    print("TESTING TRAINED MODEL")
    print("=" * 80)
    print()
    
    for idx, test_case in enumerate(TEST_PROBLEMS, 1):
        problem = test_case["problem"]
        expected = test_case["expected_answer"]
        
        print(f"Test Case {idx}:")
        print(f"Problem: {problem}")
        print(f"Expected Answer: {expected}")
        print()
        
        print("Generating solution...")
        try:
            solution = generate_solution(model, tokenizer, problem)
            
            # Extract answer
            extracted_answer = extract_answer(solution)
            
            print("Generated Solution:")
            print("-" * 80)
            print(solution)
            print("-" * 80)
            print(f"Extracted Answer: {extracted_answer}")
            print()
            
            # Check for required tags
            has_think = "<think>" in solution and "</think>" in solution
            has_answer = "<answer>" in solution and "</answer>" in solution
            
            print("Quality Checks:")
            print(f"  ✓ Has <think> tags: {has_think}")
            print(f"  ✓ Has <answer> tags: {has_answer}")
            print(f"  ✓ Length: {len(solution)} characters")
            print()
            
        except Exception as e:
            print(f"✗ Error generating solution: {e}")
            print()
            continue
        
        print("=" * 80)
        print()

def interactive_mode():
    """Interactive mode for testing custom problems."""
    
    # Load model
    model, tokenizer = load_model_and_tokenizer()
    
    print("=" * 80)
    print("INTERACTIVE MODE")
    print("=" * 80)
    print("Enter problems to test (or 'quit' to exit):")
    print()
    
    while True:
        problem = input("Problem: ").strip()
        
        if problem.lower() in ['quit', 'exit', 'q']:
            print("Exiting...")
            break
        
        if not problem:
            continue
        
        print("\nGenerating solution...")
        try:
            solution = generate_solution(model, tokenizer, problem)
            
            print("\nGenerated Solution:")
            print("-" * 80)
            print(solution)
            print("-" * 80)
            print()
            
        except Exception as e:
            print(f"Error: {e}")
            print()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the trained PPO model")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=3584,
        help="Maximum tokens to generate"
    )
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
    else:
        main()
