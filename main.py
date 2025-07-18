#!/usr/bin/env python3
"""Simple demonstration of Dendrite Transformer model."""

import torch
from model.core import DendriteForCausalLM
from transformers import AutoTokenizer


def main():
    print("=== Dendrite Transformer Demo ===\n")
    
    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name()
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "Apple MPS"
    else:
        device = torch.device("cpu")
        device_name = "CPU"
    
    print(f"Using device: {device_name}")
    
    # Initialize tokenizer
    print("\nInitializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Model configuration (small version for demo)
    config = {
        'vocab_size': tokenizer.vocab_size,
        'hidden_size': 256,  # Small for demo
        'num_hidden_layers': 4,
        'num_attention_heads': 8,
        'intermediate_size': 1024,
        'max_position_embeddings': 512,
        'use_flash_attention': False,
        'gradient_checkpointing': False,
    }
    
    # Initialize model
    print("\nInitializing model...")
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    model = DendriteForCausalLM(config)
    model = model.to(device)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test text generation
    print("\n=== Text Generation Test ===")
    test_prompts = [
        "The future of artificial intelligence",
        "Once upon a time",
        "In the beginning",
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate text
        with torch.no_grad():
            # Simple greedy generation
            generated_ids = inputs.input_ids.clone()
            max_new_tokens = 30
            
            for _ in range(max_new_tokens):
                outputs = model(generated_ids)
                logits = outputs['logits']
                
                # Get the last token's logits
                next_token_logits = logits[:, -1, :]
                
                # Greedy selection
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Check for EOS token
                if next_token_id.item() == tokenizer.eos_token_id:
                    break
                
                # Append to generated sequence
                generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
        
        # Decode generated text
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"Generated: '{generated_text}'")
    
    # Test batch processing
    print("\n=== Batch Processing Test ===")
    batch_texts = [
        "Hello world",
        "Machine learning is",
        "Python programming",
    ]
    
    # Tokenize batch
    batch_inputs = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=batch_inputs.input_ids,
            attention_mask=batch_inputs.attention_mask
        )
        logits = outputs['logits']
    
    print(f"Batch size: {len(batch_texts)}")
    print(f"Output shape: {logits.shape}")
    print(f"Vocab size matches: {logits.shape[-1] == tokenizer.vocab_size}")
    
    # Test model with different sequence lengths
    print("\n=== Variable Sequence Length Test ===")
    seq_lengths = [10, 50, 100, 200]
    
    for seq_len in seq_lengths:
        # Create dummy input
        dummy_input = torch.randint(
            0, tokenizer.vocab_size,
            (1, seq_len),
            device=device
        )
        
        with torch.no_grad():
            outputs = model(dummy_input)
            logits = outputs['logits']
        
        print(f"Seq length {seq_len}: Output shape {logits.shape}")
    
    # Memory usage summary
    if torch.cuda.is_available():
        print(f"\nGPU Memory: {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
