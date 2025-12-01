#!/usr/bin/env python3
"""
Script to push the trained PPO model to Hugging Face Hub.
"""

import os
import sys
import argparse
from pathlib import Path

def push_to_huggingface(
    model_path: str,
    repo_id: str,
    private: bool = False,
    create_pr: bool = False,
    commit_message: str = None,
    token: str = None,
):
    """
    Push model to Hugging Face Hub.
    
    Args:
        model_path: Path to the local model directory
        repo_id: Repository ID in format "username/repo-name"
        private: Whether to make the repo private
        create_pr: Whether to create a PR instead of committing directly
        commit_message: Custom commit message
        token: Hugging Face token (will use HF_TOKEN env var if not provided)
    """
    
    from huggingface_hub import HfApi, create_repo
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    print("=" * 80)
    print("PUSHING MODEL TO HUGGING FACE")
    print("=" * 80)
    print()
    
    # Validate inputs
    if not os.path.isdir(model_path):
        print(f"✗ Error: Model path does not exist: {model_path}")
        sys.exit(1)
    
    if "/" not in repo_id or len(repo_id.split("/")) != 2:
        print(f"✗ Error: Invalid repo_id format. Expected 'username/repo-name', got '{repo_id}'")
        sys.exit(1)
    
    # Get token
    if token is None:
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if token is None:
            print("✗ Error: No Hugging Face token found.")
            print("  Please set HF_TOKEN environment variable or pass --token")
            print("  You can get a token at: https://huggingface.co/settings/tokens")
            sys.exit(1)
    
    try:
        api = HfApi(token=token)
        
        # Check if model files exist
        required_files = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
        
        if missing_files:
            print(f"⚠ Warning: Missing files: {missing_files}")
            print("  The model might not work correctly without these files.")
        
        print(f"Model path: {model_path}")
        print(f"Repository: {repo_id}")
        print(f"Private: {private}")
        print(f"Create PR: {create_pr}")
        print()
        
        # Create repository if it doesn't exist
        print("Creating/checking repository...")
        try:
            repo_url = create_repo(
                repo_id=repo_id,
                private=private,
                exist_ok=True,
                token=token
            )
            print(f"✓ Repository ready: {repo_url}")
        except Exception as e:
            print(f"✗ Error creating repository: {e}")
            sys.exit(1)
        
        print()
        print("Uploading model files...")
        
        # Upload all files in the model directory
        uploaded_count = 0
        for file_path in Path(model_path).rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(model_path)
                
                # Skip unnecessary files
                if relative_path.suffix in [".bin", ".md"]:
                    continue
                
                try:
                    api.upload_file(
                        path_or_fileobj=str(file_path),
                        path_in_repo=str(relative_path),
                        repo_id=repo_id,
                        repo_type="model",
                        commit_message=commit_message or f"Upload model files",
                        create_pr=create_pr,
                    )
                    uploaded_count += 1
                    print(f"  ✓ {relative_path}")
                except Exception as e:
                    print(f"  ⚠ Error uploading {relative_path}: {e}")
        
        print()
        print("=" * 80)
        print(f"✓ Successfully uploaded {uploaded_count} files!")
        print(f"✓ Model available at: https://huggingface.co/{repo_id}")
        print("=" * 80)
        print()
        
        # Show how to use the model
        print("To use the model:")
        print()
        print("from transformers import AutoTokenizer, AutoModelForCausalLM")
        print()
        print(f"tokenizer = AutoTokenizer.from_pretrained('{repo_id}')")
        print(f"model = AutoModelForCausalLM.from_pretrained('{repo_id}', torch_dtype='auto', device_map='auto')")
        print()
        
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)

def main():
    """Main function with argument parsing."""
    
    parser = argparse.ArgumentParser(
        description="Push trained PPO model to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Push to your HF account
  python push_to_huggingface.py --repo-id username/OpenRS-PPO-v1
  
  # Push as a private repo
  python push_to_huggingface.py --repo-id username/OpenRS-PPO-v1 --private
  
  # Using custom model path
  python push_to_huggingface.py --repo-id username/my-model --model-path ./custom/path
  
  # Create a PR instead of committing directly
  python push_to_huggingface.py --repo-id username/model --create-pr

Requirements:
  1. Install huggingface_hub: pip install huggingface_hub
  2. Authenticate: huggingface-cli login
  3. Or set HF_TOKEN environment variable
        """
    )
    
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository ID in format 'username/repo-name' (e.g., 'knoveleng/OpenRS-PPO-v1')"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/rl-group10/training_scripts/open-rs/data/OpenRS-PPO_v1",
        help="Path to the local model directory"
    )
    
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private"
    )
    
    parser.add_argument(
        "--create-pr",
        action="store_true",
        help="Create a pull request instead of committing directly"
    )
    
    parser.add_argument(
        "--commit-message",
        type=str,
        help="Custom commit message"
    )
    
    parser.add_argument(
        "--token",
        type=str,
        help="Hugging Face API token (if not set, uses HF_TOKEN env var)"
    )
    
    args = parser.parse_args()
    
    # Check if huggingface_hub is installed
    try:
        import huggingface_hub
    except ImportError:
        print("✗ Error: huggingface_hub is not installed")
        print("  Install it with: pip install huggingface_hub")
        sys.exit(1)
    
    push_to_huggingface(
        model_path=args.model_path,
        repo_id=args.repo_id,
        private=args.private,
        create_pr=args.create_pr,
        commit_message=args.commit_message,
        token=args.token,
    )

if __name__ == "__main__":
    main()
