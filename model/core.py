import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Dict, Any
import math
from .positional_encoding import (
    RotaryPositionEmbedding,
    ALiBiPositionalBias,
    create_position_encoding
)


class DendriteAttention(nn.Module):
    """Dendrite-style attention mechanism with hierarchical processing."""
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_dropout: float = 0.1,
        use_flash_attention: bool = False,
        position_encoding_type: str = 'learned',
        max_position_embeddings: int = 131072,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.use_flash_attention = use_flash_attention
        
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(attention_dropout)
        
        # Position encoding
        self.position_encoding_type = position_encoding_type
        if position_encoding_type == 'rope':
            self.rotary_emb = RotaryPositionEmbedding(
                self.attention_head_size,
                max_position_embeddings
            )
        elif position_encoding_type == 'alibi':
            self.alibi = ALiBiPositionalBias(
                num_attention_heads,
                max_position_embeddings
            )
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ):
        batch_size = hidden_states.size(0)
        
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        
        if past_key_value is not None:
            key_layer = torch.cat([past_key_value[0], self.transpose_for_scores(self.key(hidden_states))], dim=2)
            value_layer = torch.cat([past_key_value[1], self.transpose_for_scores(self.value(hidden_states))], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Apply rotary position embedding if enabled
        if self.position_encoding_type == 'rope':
            query_layer, key_layer = self.rotary_emb.apply_rotary_pos_emb(
                query_layer, key_layer
            )
        
        if self.use_flash_attention and torch.cuda.is_available():
            try:
                from flash_attn import flash_attn_func
                context_layer = flash_attn_func(
                    query_layer,
                    key_layer,
                    value_layer,
                    dropout_p=self.dropout.p if self.training else 0.0,
                )
            except ImportError:
                attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
                attention_scores = attention_scores / math.sqrt(self.attention_head_size)
                
                # Apply ALiBi bias if enabled
                if self.position_encoding_type == 'alibi':
                    attention_scores = self.alibi(attention_scores)
                
                if attention_mask is not None:
                    attention_scores = attention_scores + attention_mask
                
                attention_probs = nn.functional.softmax(attention_scores, dim=-1)
                attention_probs = self.dropout(attention_probs)
                
                context_layer = torch.matmul(attention_probs, value_layer)
        else:
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            
            # Apply ALiBi bias if enabled
            if self.position_encoding_type == 'alibi':
                attention_scores = self.alibi(attention_scores)
            
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
            
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)
            attention_probs = self.dropout(attention_probs)
            
            context_layer = torch.matmul(attention_probs, value_layer)
        
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        return context_layer, (key_layer, value_layer)


class DendriteBlock(nn.Module):
    """Single Dendrite transformer block."""
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-12,
        use_flash_attention: bool = False,
        position_encoding_type: str = 'learned',
        max_position_embeddings: int = 131072,
    ):
        super().__init__()
        self.attention = DendriteAttention(
            hidden_size,
            num_attention_heads,
            attention_probs_dropout_prob,
            use_flash_attention,
            position_encoding_type,
            max_position_embeddings,
        )
        self.output = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        
        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.output_dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ):
        residual = hidden_states
        
        attention_output, present_key_value = self.attention(
            hidden_states,
            attention_mask,
            past_key_value,
        )
        attention_output = self.output(attention_output)
        attention_output = self.dropout(attention_output)
        hidden_states = self.LayerNorm(attention_output + residual)
        
        residual = hidden_states
        intermediate_output = self.intermediate(hidden_states)
        intermediate_output = F.gelu(intermediate_output)
        output = self.output_dense(intermediate_output)
        output = self.dropout(output)
        output = self.LayerNorm2(output + residual)
        
        return output, present_key_value


class DendriteTransformer(nn.Module):
    """Main Dendrite Transformer model."""
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        max_position_embeddings: int = 131072,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-12,
        use_flash_attention: bool = False,
        gradient_checkpointing: bool = False,
        position_encoding_type: str = 'learned',
    ):
        super().__init__()
        self.config = {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": num_attention_heads,
            "intermediate_size": intermediate_size,
            "max_position_embeddings": max_position_embeddings,
            "hidden_dropout_prob": hidden_dropout_prob,
            "attention_probs_dropout_prob": attention_probs_dropout_prob,
            "layer_norm_eps": layer_norm_eps,
            "use_flash_attention": use_flash_attention,
            "gradient_checkpointing": gradient_checkpointing,
            "position_encoding_type": position_encoding_type,
        }
        
        self.position_encoding_type = position_encoding_type
        
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        
        # Position encoding based on type
        if position_encoding_type in ['learned', 'sinusoidal']:
            self.position_embeddings = create_position_encoding(
                position_encoding_type,
                hidden_size,
                max_position_embeddings
            )
        else:
            # For RoPE and ALiBi, position encoding is handled in attention
            self.position_embeddings = None
        
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        
        self.layers = nn.ModuleList([
            DendriteBlock(
                hidden_size,
                num_attention_heads,
                intermediate_size,
                hidden_dropout_prob,
                attention_probs_dropout_prob,
                layer_norm_eps,
                use_flash_attention,
                position_encoding_type,
                max_position_embeddings,
            )
            for _ in range(num_hidden_layers)
        ])
        
        self.gradient_checkpointing = gradient_checkpointing
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: bool = True,
    ):
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=device)
        
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        embeddings = self.embeddings(input_ids)
        
        # Apply position embeddings if using learned or sinusoidal
        if self.position_embeddings is not None:
            if self.position_encoding_type == 'learned':
                position_embeddings = self.position_embeddings(position_ids)
            else:  # sinusoidal
                position_embeddings = self.position_embeddings(seq_length)
            hidden_states = embeddings + position_embeddings
        else:
            # For RoPE and ALiBi, position encoding is handled in attention
            hidden_states = embeddings
        
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        presents = () if use_cache else None
        
        for i, layer in enumerate(self.layers):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                
                hidden_states, present_key_value = checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    extended_attention_mask,
                    past_key_value,
                )
            else:
                hidden_states, present_key_value = layer(
                    hidden_states,
                    extended_attention_mask,
                    past_key_value,
                )
            
            if use_cache:
                presents = presents + (present_key_value,)
        
        return hidden_states, presents


class DendriteForCausalLM(nn.Module):
    """Dendrite model for causal language modeling."""
    
    def __init__(self, config):
        super().__init__()
        self.transformer = DendriteTransformer(**config)
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)
        
        self.lm_head.weight = self.transformer.embeddings.weight
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: bool = True,
    ):
        hidden_states, presents = self.transformer(
            input_ids,
            attention_mask,
            past_key_values,
            use_cache,
        )
        
        lm_logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        
        return {
            "loss": loss,
            "logits": lm_logits,
            "past_key_values": presents,
        }