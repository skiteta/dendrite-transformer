#!/usr/bin/env python3
"""Unit tests for Dendrite Transformer core components."""

import pytest
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from model.core import (
    DendriteAttention,
    DendriteBlock,
    DendriteTransformer,
    DendriteForCausalLM,
)


class TestDendriteAttention:
    """Test cases for DendriteAttention module."""
    
    def test_init(self):
        """Test DendriteAttention initialization."""
        attention = DendriteAttention(
            hidden_size=768,
            num_attention_heads=12,
            attention_dropout=0.1,
            use_flash_attention=False,
        )
        
        assert attention.hidden_size == 768
        assert attention.num_attention_heads == 12
        assert attention.attention_head_size == 64
        assert attention.all_head_size == 768
    
    def test_forward_basic(self):
        """Test basic forward pass."""
        attention = DendriteAttention(
            hidden_size=768,
            num_attention_heads=12,
            attention_dropout=0.1,
            use_flash_attention=False,
        )
        
        batch_size, seq_len = 2, 32
        hidden_states = torch.randn(batch_size, seq_len, 768)
        
        output, key_value = attention(hidden_states)
        
        assert output.shape == (batch_size, seq_len, 768)
        assert len(key_value) == 2
        assert key_value[0].shape == (batch_size, 12, seq_len, 64)
        assert key_value[1].shape == (batch_size, 12, seq_len, 64)
    
    def test_forward_with_mask(self):
        """Test forward pass with attention mask."""
        attention = DendriteAttention(
            hidden_size=768,
            num_attention_heads=12,
            attention_dropout=0.1,
            use_flash_attention=False,
        )
        
        batch_size, seq_len = 2, 32
        hidden_states = torch.randn(batch_size, seq_len, 768)
        attention_mask = torch.ones(batch_size, 1, 1, seq_len) * -10000.0
        attention_mask[:, :, :, :seq_len//2] = 0.0
        
        output, key_value = attention(hidden_states, attention_mask)
        
        assert output.shape == (batch_size, seq_len, 768)
        assert not torch.isnan(output).any()
    
    def test_forward_with_past_key_value(self):
        """Test forward pass with past key values."""
        attention = DendriteAttention(
            hidden_size=768,
            num_attention_heads=12,
            attention_dropout=0.1,
            use_flash_attention=False,
        )
        
        batch_size, seq_len = 2, 32
        hidden_states = torch.randn(batch_size, seq_len, 768)
        
        # First forward pass
        output1, key_value1 = attention(hidden_states)
        
        # Second forward pass with past key values
        new_hidden_states = torch.randn(batch_size, seq_len, 768)
        output2, key_value2 = attention(new_hidden_states, past_key_value=key_value1)
        
        assert output2.shape == (batch_size, seq_len, 768)
        assert key_value2[0].shape == (batch_size, 12, seq_len * 2, 64)
        assert key_value2[1].shape == (batch_size, 12, seq_len * 2, 64)


class TestDendriteBlock:
    """Test cases for DendriteBlock module."""
    
    def test_init(self):
        """Test DendriteBlock initialization."""
        block = DendriteBlock(
            hidden_size=768,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            layer_norm_eps=1e-12,
            use_flash_attention=False,
        )
        
        assert block.attention.hidden_size == 768
        assert block.attention.num_attention_heads == 12
        assert block.intermediate.in_features == 768
        assert block.intermediate.out_features == 3072
    
    def test_forward_basic(self):
        """Test basic forward pass."""
        block = DendriteBlock(
            hidden_size=768,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            layer_norm_eps=1e-12,
            use_flash_attention=False,
        )
        
        batch_size, seq_len = 2, 32
        hidden_states = torch.randn(batch_size, seq_len, 768)
        
        output, key_value = block(hidden_states)
        
        assert output.shape == (batch_size, seq_len, 768)
        assert len(key_value) == 2
        assert not torch.isnan(output).any()


class TestDendriteTransformer:
    """Test cases for DendriteTransformer module."""
    
    def test_init(self):
        """Test DendriteTransformer initialization."""
        transformer = DendriteTransformer(
            vocab_size=50257,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=2048,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            layer_norm_eps=1e-12,
            use_flash_attention=False,
            gradient_checkpointing=False,
        )
        
        assert transformer.embeddings.num_embeddings == 50257
        assert transformer.embeddings.embedding_dim == 768
        assert len(transformer.layers) == 12
        assert transformer.position_embeddings.num_embeddings == 2048
    
    def test_forward_basic(self):
        """Test basic forward pass."""
        transformer = DendriteTransformer(
            vocab_size=50257,
            hidden_size=768,
            num_hidden_layers=4,  # Smaller for testing
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=2048,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            layer_norm_eps=1e-12,
            use_flash_attention=False,
            gradient_checkpointing=False,
        )
        
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, 50257, (batch_size, seq_len))
        
        output, presents = transformer(input_ids)
        
        assert output.shape == (batch_size, seq_len, 768)
        assert len(presents) == 4
        assert not torch.isnan(output).any()
    
    def test_forward_with_attention_mask(self):
        """Test forward pass with attention mask."""
        transformer = DendriteTransformer(
            vocab_size=50257,
            hidden_size=768,
            num_hidden_layers=4,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=2048,
            use_flash_attention=False,
            gradient_checkpointing=False,
        )
        
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, 50257, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[:, seq_len//2:] = 0
        
        output, presents = transformer(input_ids, attention_mask)
        
        assert output.shape == (batch_size, seq_len, 768)
        assert not torch.isnan(output).any()
    
    def test_forward_with_past_key_values(self):
        """Test forward pass with past key values."""
        transformer = DendriteTransformer(
            vocab_size=50257,
            hidden_size=768,
            num_hidden_layers=4,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=2048,
            use_flash_attention=False,
            gradient_checkpointing=False,
        )
        
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, 50257, (batch_size, seq_len))
        
        # First forward pass
        output1, presents1 = transformer(input_ids)
        
        # Second forward pass with past key values
        new_input_ids = torch.randint(0, 50257, (batch_size, seq_len))
        output2, presents2 = transformer(new_input_ids, past_key_values=presents1)
        
        assert output2.shape == (batch_size, seq_len, 768)
        assert len(presents2) == 4
        assert not torch.isnan(output2).any()


class TestDendriteForCausalLM:
    """Test cases for DendriteForCausalLM module."""
    
    def test_init(self):
        """Test DendriteForCausalLM initialization."""
        config = {
            'vocab_size': 50257,
            'hidden_size': 768,
            'num_hidden_layers': 4,
            'num_attention_heads': 12,
            'intermediate_size': 3072,
            'max_position_embeddings': 2048,
            'use_flash_attention': False,
            'gradient_checkpointing': False,
        }
        
        model = DendriteForCausalLM(config)
        
        assert model.lm_head.in_features == 768
        assert model.lm_head.out_features == 50257
        assert model.lm_head.weight.shape == (50257, 768)
    
    def test_forward_basic(self):
        """Test basic forward pass."""
        config = {
            'vocab_size': 50257,
            'hidden_size': 768,
            'num_hidden_layers': 4,
            'num_attention_heads': 12,
            'intermediate_size': 3072,
            'max_position_embeddings': 2048,
            'use_flash_attention': False,
            'gradient_checkpointing': False,
        }
        
        model = DendriteForCausalLM(config)
        
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, 50257, (batch_size, seq_len))
        
        outputs = model(input_ids)
        
        assert outputs['logits'].shape == (batch_size, seq_len, 50257)
        assert outputs['past_key_values'] is not None
        assert len(outputs['past_key_values']) == 4
        assert outputs['loss'] is None
    
    def test_forward_with_labels(self):
        """Test forward pass with labels for loss computation."""
        config = {
            'vocab_size': 50257,
            'hidden_size': 768,
            'num_hidden_layers': 4,
            'num_attention_heads': 12,
            'intermediate_size': 3072,
            'max_position_embeddings': 2048,
            'use_flash_attention': False,
            'gradient_checkpointing': False,
        }
        
        model = DendriteForCausalLM(config)
        
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, 50257, (batch_size, seq_len))
        labels = input_ids.clone()
        
        outputs = model(input_ids, labels=labels)
        
        assert outputs['logits'].shape == (batch_size, seq_len, 50257)
        assert outputs['loss'] is not None
        assert isinstance(outputs['loss'].item(), float)
        assert outputs['loss'].item() > 0
    
    def test_gradient_computation(self):
        """Test that gradients can be computed."""
        config = {
            'vocab_size': 50257,
            'hidden_size': 768,
            'num_hidden_layers': 2,  # Smaller for testing
            'num_attention_heads': 12,
            'intermediate_size': 3072,
            'max_position_embeddings': 2048,
            'use_flash_attention': False,
            'gradient_checkpointing': False,
        }
        
        model = DendriteForCausalLM(config)
        
        batch_size, seq_len = 1, 16
        input_ids = torch.randint(0, 50257, (batch_size, seq_len))
        labels = input_ids.clone()
        
        outputs = model(input_ids, labels=labels)
        loss = outputs['loss']
        
        loss.backward()
        
        # Check that gradients were computed
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


class TestModelIntegration:
    """Integration tests for the complete model."""
    
    def test_model_sizes(self):
        """Test different model sizes."""
        sizes = [
            (256, 2, 4, 512),    # Mini model
            (768, 4, 12, 3072),  # Small model
        ]
        
        for hidden_size, num_layers, num_heads, intermediate_size in sizes:
            config = {
                'vocab_size': 1000,
                'hidden_size': hidden_size,
                'num_hidden_layers': num_layers,
                'num_attention_heads': num_heads,
                'intermediate_size': intermediate_size,
                'max_position_embeddings': 512,
                'use_flash_attention': False,
                'gradient_checkpointing': False,
            }
            
            model = DendriteForCausalLM(config)
            
            batch_size, seq_len = 1, 16
            input_ids = torch.randint(0, 1000, (batch_size, seq_len))
            
            outputs = model(input_ids)
            
            assert outputs['logits'].shape == (batch_size, seq_len, 1000)
            assert not torch.isnan(outputs['logits']).any()
    
    def test_gradient_checkpointing(self):
        """Test gradient checkpointing functionality."""
        config = {
            'vocab_size': 1000,
            'hidden_size': 256,
            'num_hidden_layers': 4,
            'num_attention_heads': 8,
            'intermediate_size': 1024,
            'max_position_embeddings': 512,
            'use_flash_attention': False,
            'gradient_checkpointing': True,
        }
        
        model = DendriteForCausalLM(config)
        model.train()
        
        batch_size, seq_len = 1, 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        labels = input_ids.clone()
        
        outputs = model(input_ids, labels=labels)
        loss = outputs['loss']
        
        loss.backward()
        
        # Check that gradients were computed despite checkpointing
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])