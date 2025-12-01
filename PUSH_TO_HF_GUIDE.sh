#!/bin/bash
# Quick guide to push model to Hugging Face Hub

# Step 1: Ensure you're authenticated
# Either run: huggingface-cli login
# Or set token: export HF_TOKEN="your_token_here"

# Step 2: Run the push script with your repo ID
python push_to_huggingface.py --repo-id your-username/OpenRS-PPO-v1

# Optional flags:
# --private           Make the repo private
# --create-pr         Create a pull request instead of committing
# --model-path        Custom path to model (default: ./data/OpenRS-PPO_v1)
# --token             Pass token directly (not recommended for security)

# Examples:

# 1. Public repository
# python push_to_huggingface.py --repo-id knoveleng/OpenRS-PPO-v1

# 2. Private repository
# python push_to_huggingface.py --repo-id knoveleng/OpenRS-PPO-v1 --private

# 3. With custom message
# python push_to_huggingface.py --repo-id knoveleng/OpenRS-PPO-v1 --commit-message "Upload trained model v1"

# Get your token from: https://huggingface.co/settings/tokens
