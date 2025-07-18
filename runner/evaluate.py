#!/usr/bin/env python3
"""Evaluation script for Dendrite Transformer."""

import argparse
import os
import sys
import yaml
import torch
import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from model.core import DendriteForCausalLM
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_device(config: Dict[str, Any]):
    """Setup device based on availability."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    return device


def load_model(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint.get('config', {})
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
        'max_position_embeddings': config.get('seq_len', 2048),
        'use_flash_attention': False,
        'gradient_checkpointing': False,
    }
    
    model = DendriteForCausalLM(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, tokenizer, config


def prepare_dataset(dataset_name: str, dataset_config: str, split: str = 'test'):
    """Prepare dataset for evaluation."""
    try:
        dataset = load_dataset(dataset_name, dataset_config, split=split)
    except Exception as e:
        print(f"Error loading dataset {dataset_name}:{dataset_config}: {e}")
        dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='test')
    
    return dataset


def compute_perplexity(model, dataloader, device, max_length=1024):
    """Compute perplexity on the dataset."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing perplexity"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            seq_len = input_ids.size(1)
            if seq_len > max_length:
                input_ids = input_ids[:, :max_length]
                attention_mask = attention_mask[:, :max_length]
            
            labels = input_ids.clone()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs['loss']
            
            if loss is not None:
                total_loss += loss.item() * input_ids.size(0)
                total_tokens += input_ids.size(0)
    
    if total_tokens == 0:
        return float('inf')
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return perplexity


def generate_text(model, tokenizer, prompt: str, max_length: int = 256, device: torch.device = None):
    """Generate text from a prompt."""
    model.eval()
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        past_key_values = None
        generated_ids = input_ids.clone()
        
        for _ in range(max_length):
            outputs = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True
            )
            
            logits = outputs['logits']
            past_key_values = outputs['past_key_values']
            
            next_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            if next_token_id.item() == tokenizer.eos_token_id:
                break
            
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
            input_ids = next_token_id
        
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    return generated_text


def benchmark_inference_speed(model, tokenizer, device, seq_lengths: List[int], batch_size: int = 1):
    """Benchmark inference speed for different sequence lengths."""
    model.eval()
    results = {}
    
    for seq_len in seq_lengths:
        if seq_len > model.transformer.config.get('max_position_embeddings', 2048):
            print(f"Skipping seq_len {seq_len} (exceeds max position embeddings)")
            continue
        
        dummy_input = torch.randint(
            0, tokenizer.vocab_size,
            (batch_size, seq_len),
            device=device
        )
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        import time
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        tokens_per_second = (10 * batch_size * seq_len) / (end_time - start_time)
        results[seq_len] = tokens_per_second
        
        print(f"Seq len {seq_len}: {tokens_per_second:.2f} tokens/sec")
    
    return results


def evaluate_long_context(model, tokenizer, device, scenario: str = "128k"):
    """Evaluate long context capabilities."""
    if scenario == "128k":
        max_seq_len = 131072
    elif scenario == "32k":
        max_seq_len = 32768
    else:
        max_seq_len = 2048
    
    model_max_len = model.transformer.config.get('max_position_embeddings', 2048)
    
    if max_seq_len > model_max_len:
        print(f"Skipping {scenario} test (model max len: {model_max_len})")
        return None
    
    print(f"Testing {scenario} scenario...")
    
    test_prompt = "The quick brown fox jumps over the lazy dog. " * 100
    
    try:
        generated_text = generate_text(
            model, tokenizer, test_prompt,
            max_length=min(256, max_seq_len),
            device=device
        )
        
        coherence_score = len(generated_text.split()) / 256
        
        return {
            'scenario': scenario,
            'max_seq_len': max_seq_len,
            'coherence_score': coherence_score,
            'generated_length': len(generated_text),
            'success': True
        }
    except Exception as e:
        print(f"Error in {scenario} test: {e}")
        return {
            'scenario': scenario,
            'max_seq_len': max_seq_len,
            'error': str(e),
            'success': False
        }


def main():
    parser = argparse.ArgumentParser(description='Evaluate Dendrite Transformer')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='wikitext', help='Dataset name')
    parser.add_argument('--dataset_config', type=str, default='wikitext-103-raw-v1', help='Dataset config')
    parser.add_argument('--scenario', type=str, default='32k', choices=['128k', '32k', 'standard'], help='Evaluation scenario')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation')
    parser.add_argument('--output_dir', type=str, default='./eval_results', help='Output directory')
    parser.add_argument('--compute_perplexity', action='store_true', help='Compute perplexity')
    parser.add_argument('--benchmark_speed', action='store_true', help='Benchmark inference speed')
    parser.add_argument('--test_generation', action='store_true', help='Test text generation')
    args = parser.parse_args()
    
    device = setup_device({})
    
    print(f"Loading model from {args.checkpoint}...")
    model, tokenizer, config = load_model(args.checkpoint, device)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'checkpoint': args.checkpoint,
        'scenario': args.scenario,
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'device': str(device),
    }
    
    if args.scenario in ['128k', '32k']:
        long_context_results = evaluate_long_context(model, tokenizer, device, args.scenario)
        results['long_context'] = long_context_results
    
    if args.compute_perplexity:
        print("Computing perplexity...")
        dataset = prepare_dataset(args.dataset, args.dataset_config, 'test')
        
        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=min(1024, config.get('seq_len', 2048)),
                return_tensors='pt'
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['text']
        )
        
        dataloader = DataLoader(
            tokenized_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        perplexity = compute_perplexity(model, dataloader, device)
        results['perplexity'] = perplexity
        print(f"Perplexity: {perplexity:.2f}")
    
    if args.benchmark_speed:
        print("Benchmarking inference speed...")
        seq_lengths = [128, 256, 512, 1024, 2048]
        if args.scenario == '32k':
            seq_lengths.extend([4096, 8192, 16384, 32768])
        elif args.scenario == '128k':
            seq_lengths.extend([4096, 8192, 16384, 32768, 65536, 131072])
        
        speed_results = benchmark_inference_speed(model, tokenizer, device, seq_lengths)
        results['speed_benchmark'] = speed_results
    
    if args.test_generation:
        print("Testing text generation...")
        test_prompts = [
            "The future of artificial intelligence is",
            "In a world where technology has advanced beyond our wildest dreams,",
            "The quick brown fox jumps over the lazy dog.",
        ]
        
        generation_results = []
        for prompt in test_prompts:
            generated = generate_text(model, tokenizer, prompt, max_length=128, device=device)
            generation_results.append({
                'prompt': prompt,
                'generated': generated,
                'length': len(generated)
            })
        
        results['generation_test'] = generation_results
    
    output_file = output_dir / f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    print("\nEvaluation Summary:")
    for key, value in results.items():
        if key not in ['config', 'timestamp', 'checkpoint']:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()