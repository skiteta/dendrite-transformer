#!/usr/bin/env python3
"""Minimal training script for local testing with small data."""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model.core import DendriteForCausalLM
from transformers import AutoTokenizer
import yaml
from pathlib import Path
import argparse
from tqdm import tqdm
import random


class SimpleTextDataset(Dataset):
    """Simple dataset for testing with synthetic data."""
    
    def __init__(self, tokenizer, num_samples=100, seq_length=128):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        # Generate simple synthetic text data
        templates = [
            "The weather today is {adj}.",
            "I like to {verb} in the {time}.",
            "The {noun} is {adj} and {adj}.",
            "We went to the {place} to {verb}.",
            "{name} is learning about {topic}.",
        ]
        
        adjectives = ["sunny", "beautiful", "interesting", "amazing", "wonderful", "cold", "warm"]
        verbs = ["walk", "run", "study", "read", "write", "play", "work"]
        times = ["morning", "afternoon", "evening", "night", "weekend"]
        nouns = ["book", "computer", "science", "music", "art", "language", "code"]
        places = ["park", "library", "school", "office", "beach", "mountain", "city"]
        names = ["Alice", "Bob", "Charlie", "David", "Emma", "Frank", "Grace"]
        topics = ["mathematics", "physics", "chemistry", "biology", "programming", "history", "literature"]
        
        self.texts = []
        for _ in range(num_samples):
            template = random.choice(templates)
            text = template.format(
                adj=random.choice(adjectives),
                verb=random.choice(verbs),
                time=random.choice(times),
                noun=random.choice(nouns),
                place=random.choice(places),
                name=random.choice(names),
                topic=random.choice(topics),
            )
            self.texts.append(text)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.seq_length,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].squeeze()
        attention_mask = encoded['attention_mask'].squeeze()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Create labels (shifted input_ids)
        labels = input_ids.clone()
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs['loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        # Update stats
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        avg_loss = total_loss / num_batches
        progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    return total_loss / num_batches


def evaluate(model, dataloader, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = input_ids.clone()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs['loss']
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def generate_text(model, tokenizer, prompt, max_length=50, device='cpu'):
    """Generate text from a prompt."""
    model.eval()
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    input_ids = inputs['input_ids']
    
    # Generate
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            logits = outputs['logits']
            
            # Get next token (greedy)
            next_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Stop if EOS
            if next_token_id.item() == tokenizer.eos_token_id:
                break
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token_id], dim=1)
    
    # Decode
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text


def main():
    parser = argparse.ArgumentParser(description='Mini training script for testing')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='./outputs/mini', help='Output directory')
    args = parser.parse_args()
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Initialize tokenizer
    print("\nInitializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Model configuration (very small for testing)
    config = {
        'vocab_size': tokenizer.vocab_size,
        'hidden_size': 128,
        'num_hidden_layers': 2,
        'num_attention_heads': 4,
        'intermediate_size': 512,
        'max_position_embeddings': 512,
        'use_flash_attention': False,
        'gradient_checkpointing': False,
    }
    
    print("\nModel configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Initialize model
    model = DendriteForCausalLM(config)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = SimpleTextDataset(tokenizer, num_samples=200, seq_length=64)
    eval_dataset = SimpleTextDataset(tokenizer, num_samples=50, seq_length=64)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    print("\nStarting training...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test generation before training
    print("\n=== Before Training ===")
    test_prompts = ["The weather", "I like to", "Alice is"]
    for prompt in test_prompts:
        generated = generate_text(model, tokenizer, prompt, max_length=20, device=device)
        print(f"'{prompt}' -> '{generated}'")
    
    # Train
    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Training loss: {train_loss:.4f}")
        
        # Evaluate
        eval_loss = evaluate(model, eval_loader, device)
        print(f"Evaluation loss: {eval_loss:.4f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'eval_loss': eval_loss,
            'config': config,
        }
        checkpoint_path = output_dir / f'checkpoint_epoch_{epoch + 1}.pt'
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
    
    # Test generation after training
    print("\n=== After Training ===")
    for prompt in test_prompts:
        generated = generate_text(model, tokenizer, prompt, max_length=20, device=device)
        print(f"'{prompt}' -> '{generated}'")
    
    # Save final model
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
    }
    final_path = output_dir / 'final_model.pt'
    torch.save(final_checkpoint, final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")
    
    # Save config as YAML
    config_path = output_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    print(f"Config saved to {config_path}")


if __name__ == "__main__":
    main()