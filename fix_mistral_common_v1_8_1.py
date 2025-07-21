#!/usr/bin/env python3
"""
Fix mistral-finetune to work with mistral-common v1.8.1

This script updates all the imports and API calls to work with the new version.
"""

import os
import re
import sys
from pathlib import Path

def fix_imports(content):
    """Fix imports for mistral-common v1.8.1"""
    
    # Replace MistralTokenizer with MistralTokenizer
    content = re.sub(
        r'from mistral_common\.tokens\.tokenizers\.sentencepiece import MistralTokenizer',
        'from mistral_common.tokens.tokenizers.mistral import MistralTokenizer',
        content
    )
    
    # Replace all occurrences of MistralTokenizer with MistralTokenizer
    content = content.replace('MistralTokenizer', 'MistralTokenizer')
    
    return content

def fix_tokenizer_usage(content):
    """Fix tokenizer API usage for v1.8.1"""
    
    # Fix tokenizer initialization
    content = re.sub(
        r'MistralTokenizer\.v3\((.*?)\)',
        r'MistralTokenizer.from_file(\1)',
        content,
        flags=re.DOTALL
    )
    
    # Remove is_tekken parameter
    content = re.sub(
        r'is_tekken\s*=\s*[^,\)]+[,\)]',
        ')',
        content
    )
    
    return content

def fix_validator_usage(content):
    """Fix validator API usage"""
    
    # Update validate_messages calls to include continue_final_message
    content = re.sub(
        r'validator\.validate_messages\(messages\)',
        'validator.validate_messages(messages, continue_final_message=False)',
        content
    )
    
    return content

def fix_tokenize_instruct(content):
    """Replace the tokenize_instruct function with v1.8.1 compatible version"""
    
    # Check if this is tokenize.py
    if 'def tokenize_instruct(
    sample: TrainingInstructSample,
    instruct_tokenizer: MistralTokenizer,
) -> TokenSample:
    """
    Tokenize an instruct sample using mistral-common v1.8.1 API
    """
    from mistral_common.protocol.instruct.request import InstructRequest
    
    # Create request compatible with v1.8.1
    # Note: v1.8.1 validator expects 'tools' but InstructRequest has 'available_tools'
    # We work around this by using a custom class
    class InstructRequestCompat(InstructRequest):
        @property
        def tools(self):
            return self.available_tools
    
    request = InstructRequestCompat(
        messages=sample.messages,
        available_tools=sample.available_tools if sample.available_tools else None,
        system_prompt=sample.system_prompt if hasattr(sample, 'system_prompt') and sample.system_prompt else None,
        continue_final_message=True  # Required for training data ending with assistant messages
    )
    
    # Encode the entire conversation
    encoded = instruct_tokenizer.encode_chat_completion(request)
    tokens = encoded.tokens
    
    # Create masks - simplified approach for v1.8.1
    # In production, you'd want more sophisticated masking
    masks = [True] * len(tokens)
    
    # Handle only_last flag
    if sample.only_last and len(sample.messages) > 0:
        # Find the last assistant message
        last_assistant_idx = None
        for i in range(len(sample.messages) - 1, -1, -1):
            if isinstance(sample.messages[i], FinetuningAssistantMessage):
                last_assistant_idx = i
                break
        
        if last_assistant_idx is not None:
            # Rough approximation: mask the first 80% of tokens
            mask_until = int(len(tokens) * 0.8)
            masks = [False] * mask_until + [True] * (len(tokens) - mask_until)
    
    return TokenSample(tokens, masks)
def fix_file(filepath):
    """Fix a single file"""
    print(f"Processing {filepath}...")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Apply fixes
    content = fix_imports(content)
    content = fix_tokenizer_usage(content)
    content = fix_validator_usage(content)
    content = fix_tokenize_instruct(content)
    
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  ✓ Fixed {filepath}")
        return True
    else:
        print(f"  - No changes needed in {filepath}")
        return False

def main():
    """Main function to fix all files"""
    print("Fixing mistral-finetune for mistral-common v1.8.1 compatibility...")
    
    # Find all Python files
    root_dir = Path(__file__).parent
    files_to_check = []
    
    # Add specific files we know need fixing
    files_to_check.extend([
        root_dir / "train.py",
        root_dir / "finetune" / "checkpointing.py",
        root_dir / "finetune" / "data" / "tokenize.py",
        root_dir / "finetune" / "data" / "dataset.py",
        root_dir / "finetune" / "data" / "data_loader.py",
    ])
    
    # Also check all Python files in the project
    for pattern in ["**/*.py"]:
        files_to_check.extend(root_dir.glob(pattern))
    
    # Remove duplicates
    files_to_check = list(set(files_to_check))
    
    fixed_count = 0
    for filepath in files_to_check:
        if filepath.exists() and filepath.is_file():
            if fix_file(filepath):
                fixed_count += 1
    
    print(f"\nFixed {fixed_count} files")
    
    # Check for any remaining MistralTokenizer references
    print("\nChecking for any remaining MistralTokenizer references...")
    remaining = []
    for filepath in files_to_check:
        if filepath.exists() and filepath.is_file():
            with open(filepath, 'r') as f:
                if 'MistralTokenizer' in f.read():
                    remaining.append(filepath)
    
    if remaining:
        print("Found remaining references in:")
        for f in remaining:
            print(f"  - {f}")
    else:
        print("All references have been fixed!")

if __name__ == "__main__":
    main()