#!/usr/bin/env python3
"""Comprehensive benchmark suite for Dendrite Transformer."""

import torch
import time
import json
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import psutil
import GPUtil

from transformers import AutoTokenizer
from datasets import load_dataset
from rouge_score import rouge_scorer
import sys
sys.path.append(str(Path(__file__).parent.parent))

from model.core import DendriteForCausalLM


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    model_name: str
    model_config: Dict
    timestamp: str
    system_info: Dict
    inference_speed: Dict
    memory_usage: Dict
    perplexity: Optional[float] = None
    generation_quality: Optional[Dict] = None
    attention_pattern: Optional[Dict] = None


class DendriteBenchmark:
    """Comprehensive benchmark suite for Dendrite Transformer."""
    
    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        
        # Setup device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self._load_model()
        
    def _load_model(self):
        """Load model and tokenizer."""
        print(f"Loading model from {self.model_path}")
        
        if self.model_path.endswith('.pt'):
            # Load from checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            config = checkpoint.get('config', {})
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.get('tokenizer_name', 'gpt2')
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            model_config = {
                'vocab_size': self.tokenizer.vocab_size,
                'hidden_size': config.get('hidden_size', 768),
                'num_hidden_layers': config.get('num_layers', 12),
                'num_attention_heads': config.get('num_heads', 12),
                'intermediate_size': config.get('intermediate_size', 3072),
                'max_position_embeddings': config.get('seq_len', 2048),
                'position_encoding_type': config.get('position_encoding_type', 'learned'),
            }
            
            self.model = DendriteForCausalLM(model_config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Create fresh model with config
            with open(self.model_path, 'r') as f:
                config = json.load(f)
            
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = DendriteForCausalLM(config)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Store config
        self.model_config = self.model.transformer.config
        
        print(f"Model loaded successfully")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def get_system_info(self) -> Dict:
        """Get system information."""
        info = {
            'platform': sys.platform,
            'python_version': sys.version,
            'torch_version': torch.__version__,
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else None,
            'memory_total': psutil.virtual_memory().total / 1e9,
            'device': str(self.device),
        }
        
        if torch.cuda.is_available():
            info['cuda_version'] = torch.version.cuda
            info['gpu_name'] = torch.cuda.get_device_name()
            info['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory / 1e9
            
        return info
    
    def benchmark_inference_speed(
        self, 
        seq_lengths: List[int] = [128, 512, 1024, 2048, 4096],
        batch_sizes: List[int] = [1, 2, 4, 8],
        num_runs: int = 10
    ) -> Dict:
        """Benchmark inference speed for different configurations."""
        print("\n=== Inference Speed Benchmark ===")
        results = {}
        
        for batch_size in batch_sizes:
            results[f"batch_{batch_size}"] = {}
            
            for seq_len in seq_lengths:
                if seq_len > self.model_config.get('max_position_embeddings', 2048):
                    continue
                
                # Create dummy input
                input_ids = torch.randint(
                    0, self.tokenizer.vocab_size,
                    (batch_size, seq_len),
                    device=self.device
                )
                
                # Warmup
                for _ in range(3):
                    with torch.no_grad():
                        _ = self.model(input_ids)
                
                # Measure
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                times = []
                for _ in range(num_runs):
                    start_time = time.time()
                    
                    with torch.no_grad():
                        _ = self.model(input_ids)
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    end_time = time.time()
                    times.append(end_time - start_time)
                
                # Calculate metrics
                avg_time = np.mean(times)
                std_time = np.std(times)
                tokens_per_second = (batch_size * seq_len) / avg_time
                
                results[f"batch_{batch_size}"][f"seq_{seq_len}"] = {
                    'avg_time': avg_time,
                    'std_time': std_time,
                    'tokens_per_second': tokens_per_second,
                    'latency_ms': avg_time * 1000,
                }
                
                print(f"Batch {batch_size}, Seq {seq_len}: "
                      f"{tokens_per_second:.2f} tokens/s, "
                      f"{avg_time*1000:.2f}ms latency")
        
        return results
    
    def benchmark_memory_usage(
        self,
        seq_lengths: List[int] = [128, 512, 1024, 2048, 4096]
    ) -> Dict:
        """Benchmark memory usage for different sequence lengths."""
        print("\n=== Memory Usage Benchmark ===")
        results = {}
        
        for seq_len in seq_lengths:
            if seq_len > self.model_config.get('max_position_embeddings', 2048):
                continue
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            # Create input
            input_ids = torch.randint(
                0, self.tokenizer.vocab_size,
                (1, seq_len),
                device=self.device
            )
            
            # Measure memory
            if torch.cuda.is_available():
                start_memory = torch.cuda.memory_allocated()
                
                with torch.no_grad():
                    _ = self.model(input_ids)
                
                end_memory = torch.cuda.memory_allocated()
                peak_memory = torch.cuda.max_memory_allocated()
                
                results[f"seq_{seq_len}"] = {
                    'allocated_mb': (end_memory - start_memory) / 1e6,
                    'peak_mb': peak_memory / 1e6,
                    'model_mb': start_memory / 1e6,
                }
                
                print(f"Seq {seq_len}: Peak {peak_memory/1e6:.2f}MB")
            else:
                # CPU/MPS memory tracking is limited
                results[f"seq_{seq_len}"] = {
                    'status': 'Memory tracking not available on this device'
                }
        
        return results
    
    def benchmark_perplexity(
        self,
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-103-raw-v1",
        split: str = "test",
        max_samples: int = 100,
        max_length: int = 1024
    ) -> float:
        """Calculate perplexity on a dataset."""
        print("\n=== Perplexity Benchmark ===")
        
        # Load dataset
        dataset = load_dataset(dataset_name, dataset_config, split=split)
        
        total_loss = 0
        total_tokens = 0
        
        for i, sample in enumerate(dataset):
            if i >= max_samples:
                break
            
            text = sample['text']
            if not text or len(text.strip()) == 0:
                continue
            
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                max_length=max_length,
                truncation=True,
                padding=False
            ).to(self.device)
            
            if inputs.input_ids.size(1) < 2:
                continue
            
            # Calculate loss
            with torch.no_grad():
                outputs = self.model(
                    input_ids=inputs.input_ids,
                    labels=inputs.input_ids
                )
                loss = outputs['loss']
            
            if loss is not None:
                total_loss += loss.item() * inputs.input_ids.size(1)
                total_tokens += inputs.input_ids.size(1)
        
        # Calculate perplexity
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = np.exp(avg_loss)
        
        print(f"Perplexity: {perplexity:.2f}")
        return perplexity
    
    def benchmark_generation_quality(
        self,
        prompts: Optional[List[str]] = None,
        max_new_tokens: int = 128,
        num_beams: int = 1,
        temperature: float = 1.0,
        top_p: float = 0.9
    ) -> Dict:
        """Benchmark text generation quality."""
        print("\n=== Generation Quality Benchmark ===")
        
        if prompts is None:
            prompts = [
                "The future of artificial intelligence is",
                "Climate change is one of the most pressing issues",
                "In the field of quantum computing,",
                "The human brain is remarkably complex",
                "Space exploration has always fascinated",
            ]
        
        results = []
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        for prompt in prompts:
            print(f"\nPrompt: '{prompt}'")
            
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
            
            # Generate
            start_time = time.time()
            
            with torch.no_grad():
                generated_ids = self._generate(
                    inputs.input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    num_beams=num_beams
                )
            
            generation_time = time.time() - start_time
            
            # Decode
            generated_text = self.tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True
            )
            
            # Calculate metrics
            generated_only = generated_text[len(prompt):].strip()
            
            result = {
                'prompt': prompt,
                'generated': generated_only,
                'generation_time': generation_time,
                'tokens_generated': len(generated_ids[0]) - len(inputs.input_ids[0]),
                'tokens_per_second': (len(generated_ids[0]) - len(inputs.input_ids[0])) / generation_time
            }
            
            results.append(result)
            print(f"Generated: '{generated_only[:100]}...'")
            print(f"Speed: {result['tokens_per_second']:.2f} tokens/s")
        
        return {
            'generations': results,
            'avg_tokens_per_second': np.mean([r['tokens_per_second'] for r in results])
        }
    
    def _generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 0.9,
        num_beams: int = 1
    ) -> torch.Tensor:
        """Simple generation function."""
        generated = input_ids.clone()
        past_key_values = None
        
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True
                )
            
            logits = outputs['logits']
            past_key_values = outputs['past_key_values']
            
            # Get next token logits
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-p sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits.scatter_(-1, indices_to_remove, float('-inf'))
            
            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            generated = torch.cat([generated, next_token], dim=1)
            input_ids = next_token
            
            # Stop if EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        return generated
    
    def benchmark_attention_patterns(self, seq_len: int = 128) -> Dict:
        """Analyze attention patterns for different position encodings."""
        print("\n=== Attention Pattern Analysis ===")
        
        # Create input
        text = "The quick brown fox jumps over the lazy dog. " * (seq_len // 10)
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=seq_len,
            truncation=True,
            padding='max_length'
        ).to(self.device)
        
        # Get attention weights
        # Note: This would require modifying the model to return attention weights
        # For now, we'll just report the position encoding type
        
        result = {
            'position_encoding_type': self.model_config.get('position_encoding_type', 'learned'),
            'max_position_embeddings': self.model_config.get('max_position_embeddings', 2048),
            'num_attention_heads': self.model_config.get('num_attention_heads', 12),
        }
        
        print(f"Position encoding: {result['position_encoding_type']}")
        print(f"Max positions: {result['max_position_embeddings']}")
        
        return result
    
    def run_full_benchmark(self) -> BenchmarkResult:
        """Run complete benchmark suite."""
        print("="*80)
        print("DENDRITE TRANSFORMER BENCHMARK SUITE")
        print("="*80)
        
        # Get system info
        system_info = self.get_system_info()
        
        # Run benchmarks
        inference_speed = self.benchmark_inference_speed()
        memory_usage = self.benchmark_memory_usage()
        perplexity = self.benchmark_perplexity(max_samples=50)
        generation_quality = self.benchmark_generation_quality()
        attention_pattern = self.benchmark_attention_patterns()
        
        # Create result
        result = BenchmarkResult(
            model_name=self.model_config.get('model_name', 'dendrite'),
            model_config=self.model_config,
            timestamp=datetime.now().isoformat(),
            system_info=system_info,
            inference_speed=inference_speed,
            memory_usage=memory_usage,
            perplexity=perplexity,
            generation_quality=generation_quality,
            attention_pattern=attention_pattern
        )
        
        return result
    
    def save_results(self, result: BenchmarkResult, output_path: str):
        """Save benchmark results to JSON."""
        with open(output_path, 'w') as f:
            json.dump(asdict(result), f, indent=2)
        print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark Dendrite Transformer')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint or config')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cuda, mps, cpu)')
    parser.add_argument('--output', type=str, default='benchmark_results.json', help='Output file path')
    parser.add_argument('--quick', action='store_true', help='Run quick benchmark (fewer samples)')
    args = parser.parse_args()
    
    # Create benchmark
    benchmark = DendriteBenchmark(args.model, args.device)
    
    # Run benchmark
    if args.quick:
        # Quick benchmark with fewer samples
        benchmark.benchmark_inference_speed(
            seq_lengths=[128, 512, 1024],
            batch_sizes=[1, 4],
            num_runs=5
        )
        benchmark.benchmark_generation_quality(
            prompts=["The future of AI is"],
            max_new_tokens=50
        )
        
        # Save quick results
        result = BenchmarkResult(
            model_name="dendrite-quick",
            model_config=benchmark.model_config,
            timestamp=datetime.now().isoformat(),
            system_info=benchmark.get_system_info(),
            inference_speed={},
            memory_usage={}
        )
    else:
        # Full benchmark
        result = benchmark.run_full_benchmark()
    
    # Save results
    benchmark.save_results(result, args.output)
    
    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"Model: {result.model_name}")
    print(f"Device: {result.system_info['device']}")
    if result.perplexity:
        print(f"Perplexity: {result.perplexity:.2f}")
    if result.generation_quality:
        print(f"Avg generation speed: {result.generation_quality['avg_tokens_per_second']:.2f} tokens/s")


if __name__ == "__main__":
    main()