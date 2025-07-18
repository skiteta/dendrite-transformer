#!/usr/bin/env python3
"""Test script for different position encoding types."""

import torch
from model.core import DendriteForCausalLM
from transformers import AutoTokenizer


def test_position_encoding(encoding_type: str):
    """Test a specific position encoding type."""
    print(f"\n{'='*50}")
    print(f"Testing {encoding_type.upper()} Position Encoding")
    print('='*50)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Model configuration
    config = {
        'vocab_size': tokenizer.vocab_size,
        'hidden_size': 256,
        'num_hidden_layers': 2,
        'num_attention_heads': 8,
        'intermediate_size': 1024,
        'max_position_embeddings': 2048,
        'use_flash_attention': False,
        'gradient_checkpointing': False,
        'position_encoding_type': encoding_type,
    }
    
    print(f"Model config: {config}")
    
    # Initialize model
    model = DendriteForCausalLM(config)
    model = model.to(device)
    model.eval()
    
    # Test inputs
    test_texts = [
        "The quick brown fox",
        "In the beginning was the Word",
        "To be or not to be, that is the question",
    ]
    
    for text in test_texts:
        print(f"\nInput: '{text}'")
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(inputs.input_ids)
            logits = outputs['logits']
            
            # Get predictions
            predictions = torch.argmax(logits, dim=-1)
            
            # Generate a few tokens
            generated_ids = inputs.input_ids.clone()
            for _ in range(10):
                outputs = model(generated_ids)
                next_token_logits = outputs['logits'][:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                if next_token_id.item() == tokenizer.eos_token_id:
                    break
                    
                generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
            
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print(f"Generated: '{generated_text}'")
    
    # Test long sequence handling
    print("\n--- Long Sequence Test ---")
    seq_lengths = [128, 512, 1024]
    
    for seq_len in seq_lengths:
        if seq_len > config['max_position_embeddings']:
            continue
            
        dummy_input = torch.randint(0, tokenizer.vocab_size, (1, seq_len), device=device)
        
        try:
            with torch.no_grad():
                outputs = model(dummy_input)
                print(f"Seq len {seq_len}: Success! Output shape: {outputs['logits'].shape}")
        except Exception as e:
            print(f"Seq len {seq_len}: Failed - {str(e)}")
    
    # Memory usage
    if torch.cuda.is_available():
        print(f"\nGPU Memory: {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB")
    
    return model


def compare_encodings():
    """Compare different position encoding types."""
    encoding_types = ['learned', 'sinusoidal', 'rope', 'alibi']
    
    print("Comparing Position Encoding Types")
    print("="*80)
    
    results = {}
    
    for encoding_type in encoding_types:
        try:
            model = test_position_encoding(encoding_type)
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            results[encoding_type] = {
                'total_params': total_params,
                'trainable_params': trainable_params,
                'status': 'Success'
            }
            
        except Exception as e:
            results[encoding_type] = {
                'status': f'Failed: {str(e)}'
            }
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for encoding_type, result in results.items():
        print(f"\n{encoding_type.upper()}:")
        if result['status'] == 'Success':
            print(f"  Total parameters: {result['total_params']:,}")
            print(f"  Trainable parameters: {result['trainable_params']:,}")
        else:
            print(f"  {result['status']}")


if __name__ == "__main__":
    compare_encodings()