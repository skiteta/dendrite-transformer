#!/usr/bin/env python3
"""Training script for NVIDIA RTX A5500 using CUDA backend."""

import argparse
import os
import sys
import yaml
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

sys.path.append(str(Path(__file__).parent.parent))

from model.core import DendriteForCausalLM
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_cuda_environment(config: Dict[str, Any]):
    """Setup CUDA environment and check availability."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this system")
    
    torch.cuda.empty_cache()
    
    device_id = config.get('cuda_device', 0)
    device = torch.device(f'cuda:{device_id}')
    
    print(f"CUDA Device: {torch.cuda.get_device_name(device_id)}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(device_id).total_memory / 1e9:.2f} GB")
    
    if config.get('deterministic', False):
        torch.manual_seed(config.get('seed', 42))
        torch.cuda.manual_seed_all(config.get('seed', 42))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    
    return device


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
        remove_columns=['text'],
        num_proc=config.get('num_proc', 4)
    )
    
    return tokenized_dataset


def setup_lora(model, config: Dict[str, Any]):
    """Setup LoRA if configured."""
    if not config.get('lora'):
        return model
    
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        
        lora_config = LoraConfig(
            r=config['lora'].get('r', 16),
            lora_alpha=config['lora'].get('alpha', 32),
            target_modules=['query', 'key', 'value', 'output', 'output_dense'],
            lora_dropout=config['lora'].get('dropout', 0.05),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
    except ImportError:
        print("PEFT not available, training without LoRA")
    
    return model


def train_step(model, batch, optimizer, config, scaler=None):
    """Single training step with mixed precision support."""
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    
    labels = input_ids.clone()
    
    if config.get('precision') == 'fp16' and scaler is not None:
        with autocast():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs['loss']
        
        scaler.scale(loss).backward()
        
        if config.get('gradient_clip', 1.0) > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
    else:
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
    parser = argparse.ArgumentParser(description='Train Dendrite Transformer on RTX A5500')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = setup_cuda_environment(config)
    
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
        'use_flash_attention': config.get('flash_attention', True),
        'gradient_checkpointing': config.get('gradient_checkpointing', True),
        'position_encoding_type': config.get('position_encoding_type', 'learned'),
    }
    
    model = DendriteForCausalLM(model_config)
    model = setup_lora(model, config)
    model = model.to(device)
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Resumed from checkpoint: {args.resume}")
    
    optimizer_name = config.get('optimizer', 'adamw')
    if optimizer_name == 'adamw_8bit':
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                model.parameters(),
                lr=config.get('learning_rate', 1e-4),
                betas=(config.get('beta1', 0.9), config.get('beta2', 0.999)),
                weight_decay=config.get('weight_decay', 0.01)
            )
        except ImportError:
            print("bitsandbytes not available, falling back to regular AdamW")
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.get('learning_rate', 1e-4),
                weight_decay=config.get('weight_decay', 0.01)
            )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            betas=(config.get('beta1', 0.9), config.get('beta2', 0.999)),
            weight_decay=config.get('weight_decay', 0.01)
        )
    
    scaler = GradScaler() if config.get('precision') == 'fp16' else None
    
    train_dataset = prepare_dataset(config, tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 4),
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
        drop_last=True
    )
    
    model.train()
    global_step = 0
    total_loss = 0
    
    num_epochs = config.get('num_epochs', 1)
    log_interval = config.get('log_interval', 10)
    save_interval = config.get('save_interval', 1000)
    
    print(f"Starting training for {num_epochs} epochs...")
    
    if config.get('compile', False) and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_steps = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            loss = train_step(model, batch, optimizer, config, scaler)
            
            total_loss += loss
            epoch_loss += loss
            global_step += 1
            epoch_steps += 1
            
            if global_step % log_interval == 0:
                avg_loss = total_loss / log_interval
                gpu_memory = torch.cuda.max_memory_allocated(device) / 1e9
                progress_bar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'gpu_mem': f'{gpu_memory:.2f}GB'
                })
                total_loss = 0
            
            if global_step % save_interval == 0:
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'global_step': global_step,
                    'epoch': epoch,
                    'config': config,
                }
                if scaler is not None:
                    checkpoint['scaler_state_dict'] = scaler.state_dict()
                
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
    if scaler is not None:
        final_checkpoint['scaler_state_dict'] = scaler.state_dict()
    
    final_path = output_dir / 'final_model.pt'
    torch.save(final_checkpoint, final_path)
    print(f"Training completed. Final model saved to {final_path}")


if __name__ == "__main__":
    main()