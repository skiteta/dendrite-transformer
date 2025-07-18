#!/usr/bin/env python3
"""Training script for Apple M3 Ultra using MLX backend."""

import argparse
import os
import sys
import yaml
import torch
import torch.backends.mps
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

sys.path.append(str(Path(__file__).parent.parent))

from model.core import DendriteForCausalLM
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def check_mlx_availability():
    """Check if MLX is available and fallback to MPS if needed."""
    try:
        import mlx
        print(f"MLX version: {mlx.__version__}")
        return True
    except ImportError:
        print("MLX not available, using MPS backend")
        if torch.backends.mps.is_available():
            print("MPS backend is available")
            return False
        else:
            raise RuntimeError("Neither MLX nor MPS backend available on this system")


def prepare_dataset(config: Dict[str, Any], tokenizer):
    """Prepare dataset for training."""
    dataset = load_dataset(
        config.get('dataset_name', 'wikitext'),
        config.get('dataset_config', 'wikitext-103-raw-v1'),
        split='train'
    )
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=config['seq_len'],
            return_tensors='pt'
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )
    
    return tokenized_dataset


def train_step(model, batch, optimizer, config):
    """Single training step."""
    input_ids = batch['input_ids'].squeeze()
    attention_mask = batch['attention_mask'].squeeze()
    
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)
    
    labels = input_ids.clone()
    
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    
    loss = outputs['loss']
    
    optimizer.zero_grad()
    loss.backward()
    
    if config.get('gradient_clip', 1.0) > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
    
    optimizer.step()
    
    return loss.item()


def main():
    parser = argparse.ArgumentParser(description='Train Dendrite Transformer on M3 Ultra')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    args = parser.parse_args()
    
    config = load_config(args.config)
    use_mlx = check_mlx_availability()
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        torch.set_num_threads(config.get('num_threads', 12))
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    print(f"Config: {config}")
    
    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.get('tokenizer_name', 'gpt2'),
        use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    model_config = {
        'vocab_size': tokenizer.vocab_size,
        'hidden_size': config.get('hidden_size', 768),
        'num_hidden_layers': config.get('num_layers', 12),
        'num_attention_heads': config.get('num_heads', 12),
        'intermediate_size': config.get('intermediate_size', 3072),
        'max_position_embeddings': config['seq_len'],
        'use_flash_attention': False,  # Flash attention not supported on MPS
        'gradient_checkpointing': config.get('gradient_checkpointing', False),
        'position_encoding_type': config.get('position_encoding_type', 'learned'),
    }
    
    model = DendriteForCausalLM(model_config)
    model = model.to(device)
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Resumed from checkpoint: {args.resume}")
    
    optimizer_name = config.get('optimizer', 'adamw')
    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            betas=(config.get('beta1', 0.9), config.get('beta2', 0.999)),
            weight_decay=config.get('weight_decay', 0.01)
        )
    elif optimizer_name == 'lion':
        try:
            from lion_pytorch import Lion
            optimizer = Lion(
                model.parameters(),
                lr=config.get('learning_rate', 1e-4),
                weight_decay=config.get('weight_decay', 0.01)
            )
        except ImportError:
            print("Lion optimizer not available, falling back to AdamW")
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.get('learning_rate', 1e-4),
                weight_decay=config.get('weight_decay', 0.01)
            )
    
    train_dataset = prepare_dataset(config, tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 1),
        shuffle=True,
        num_workers=0  # MPS doesn't support multiprocessing
    )
    
    model.train()
    global_step = 0
    total_loss = 0
    
    num_epochs = config.get('num_epochs', 1)
    log_interval = config.get('log_interval', 10)
    save_interval = config.get('save_interval', 1000)
    
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_steps = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            loss = train_step(model, batch, optimizer, config)
            
            total_loss += loss
            epoch_loss += loss
            global_step += 1
            epoch_steps += 1
            
            if global_step % log_interval == 0:
                avg_loss = total_loss / log_interval
                progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
                total_loss = 0
            
            if global_step % save_interval == 0:
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'global_step': global_step,
                    'epoch': epoch,
                    'config': config,
                }
                checkpoint_path = output_dir / f'checkpoint_step_{global_step}.pt'
                torch.save(checkpoint, checkpoint_path)
                print(f"\nSaved checkpoint to {checkpoint_path}")
            
            if config.get('max_steps') and global_step >= config['max_steps']:
                break
        
        avg_epoch_loss = epoch_loss / epoch_steps
        print(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
        
        if config.get('max_steps') and global_step >= config['max_steps']:
            break
    
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step,
        'config': config,
    }
    final_path = output_dir / 'final_model.pt'
    torch.save(final_checkpoint, final_path)
    print(f"Training completed. Final model saved to {final_path}")


if __name__ == "__main__":
    main()