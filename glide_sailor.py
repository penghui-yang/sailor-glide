from typing import Optional

import time
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from sailor_custom import Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2MLP,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
    apply_rotary_pos_emb,
)
from dataclasses import dataclass
from transformers.utils import ModelOutput
# from models.mixin import PretrainedModelParallelPreSplitMixin
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask

@dataclass
class CausalLMOutputWithPast(ModelOutput):
    llm_loss: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None

class GlideAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = Qwen2RotaryEmbedding(config=self.config)
        self.K_Cache = None
        self.V_Cache = None
        self.answer_K_Cache = None
        self.answer_V_Cache = None
        self.prefix_lens = None
        self.layer_idx = layer_idx
        self.softmax_scale = 1 / (self.head_dim ** 0.5)

        # These two variables will be reset by Qwen2GlideDecoderLayer when using `generate` function
        self.max_len = 512  
        self.range_indices = torch.arange(self.max_len)

        self.set_torch_mask()
    
    def set_torch_mask(self, max_len=4096, block_size=4):
        q_idx = torch.arange(max_len).view(-1, 1)
        kv_idx = torch.arange(max_len).view(1, -1)
        self.torch_mask = q_idx // block_size > kv_idx // block_size
        self.torch_mask = self.torch_mask.cuda()
        self.torch_mask[:4, :4] = True

    def forward(
        self,
        hidden_states,
        position_embeddings,
        cache_lens=None,
        flex_attn=None,
        exec_type="training",
        k_cache=None,
        v_cache=None,
        llm_kv_len=None,
        tree_mask=None,
    ):
        if exec_type in ["prefill", "sa_prefill"]:
            y = self.prefill(hidden_states, position_embeddings)
        elif exec_type == "sa_training":
            y = self.sa_training(hidden_states, position_embeddings)
        elif exec_type == "sa_decoding":
            y = self.decoding(hidden_states, position_embeddings, cache_lens, K_Cache=None, V_Cache=None)
        elif exec_type in ["decoding", "ca_decoding", "ca_prefill"]:
            y = self.decoding(hidden_states, position_embeddings, cache_lens, k_cache, v_cache, llm_kv_len)
        elif exec_type in ["sa_tree_decoding"]:
            y = self.tree_decoding(hidden_states, position_embeddings, cache_lens, None, None, llm_kv_len, tree_mask)
        elif exec_type in ["ca_tree_decoding"]:
            y = self.tree_decoding(hidden_states, position_embeddings, cache_lens, k_cache, v_cache, llm_kv_len, tree_mask)
        elif exec_type == "ca_training":
            y = self.glide_cross_attn_training(hidden_states, position_embeddings, k_cache, v_cache)
        else:
            raise ValueError(f"Unknown inference_type: {exec_type}")
        return y

    def glide_cross_attn_training(
            self,
            hidden_states,
            position_embeddings,
            k_cache,
            v_cache,
            ):
        k_cache = k_cache.clone().requires_grad_(True)
        v_cache = v_cache.clone().requires_grad_(True)

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = k_cache.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = v_cache.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, _ = apply_rotary_pos_emb(query_states, query_states, cos, sin, unsqueeze_dim=1)

        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
        mask = self.torch_mask[None, None, :q_len, :q_len]
        scores = torch.matmul(query_states, key_states.transpose(3, 2)) / (key_states.size(-1) ** 0.5)
        scores = scores.masked_fill(~mask, float('-inf'))
        attn_weights = F.softmax(scores.float(), dim=-1)
        attn_output = torch.matmul(attn_weights.to(value_states.dtype), value_states)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output
    
    def sa_training(
            self,
            hidden_states,
            position_embeddings,
            ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=1)

        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
        mask = self.torch_mask[None, None, :q_len, :q_len]
        scores = torch.matmul(query_states, key_states.transpose(3, 2)) / (key_states.size(-1) ** 0.5)
        scores = scores.masked_fill(~mask, float('-inf'))
        attn_weights = F.softmax(scores.float(), dim=-1)
        attn_output = torch.matmul(attn_weights.to(value_states.dtype), value_states)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output

    def prefill(
            self,
            hidden_states,
            position_embeddings,
            ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=1)
        self.K_Cache = query_states.new_zeros((bsz, self.num_key_value_heads, q_len + self.max_len, self.head_dim))
        self.V_Cache = query_states.new_zeros((bsz, self.num_key_value_heads, q_len + self.max_len, self.head_dim))
        self.K_Cache[:, :, :q_len, :] = key_states
        self.V_Cache[:, :, :q_len, :] = value_states

        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1).transpose(3, 2)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
        mask = torch.triu(torch.ones(q_len, q_len, device=query_states.device), diagonal=1).bool()
        scores = torch.matmul(query_states, key_states) * self.softmax_scale
        scores = scores.masked_fill(mask, float('-inf'))
        attn_weights = F.softmax(scores.float(), dim=-1)
        attn_output = torch.matmul(attn_weights.to(value_states.dtype), value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        self.range_indices = self.range_indices.to(self.K_Cache.device)

        attn_output = self.o_proj(attn_output)

        return attn_output
    
    def decoding(
            self,
            hidden_states,
            position_embeddings,
            cache_lens,
            K_Cache,
            V_Cache,
            llm_kv_len=None
            ):

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=1)

        if K_Cache is None:
            self.K_Cache[:, :, cache_lens : cache_lens + q_len, :] = key_states
            self.V_Cache[:, :, cache_lens : cache_lens + q_len, :] = value_states
            K_total = torch.cat((self.K_Cache[:, :, :cache_lens, :], key_states), dim=2)
            V_total = torch.cat((self.V_Cache[:, :, :cache_lens, :], value_states), dim=2)
            K_total = K_total.repeat_interleave(self.num_key_value_groups, dim=1)
            V_total = V_total.repeat_interleave(self.num_key_value_groups, dim=1)
            mask = torch.triu(torch.ones(q_len, q_len, device=query_states.device), diagonal=1).bool()
            total_mask = torch.cat((torch.zeros((q_len, cache_lens), device=query_states.device, dtype=bool), mask), dim=1)
            scores = torch.matmul(query_states, K_total.transpose(3, 2)) / (K_total.size(-1) ** 0.5)
            scores = scores.masked_fill(total_mask, float('-inf'))
            attn_weights = F.softmax(scores.float(), dim=-1)
            attn_output = torch.matmul(attn_weights.to(V_total.dtype), V_total)
        else:
            k_cache = K_Cache[:, :, :llm_kv_len, :].repeat_interleave(self.num_key_value_groups, dim=1)
            v_cache = V_Cache[:, :, :llm_kv_len, :].repeat_interleave(self.num_key_value_groups, dim=1)
            scores = torch.matmul(query_states, k_cache.transpose(3, 2)) / (k_cache.size(-1) ** 0.5)
            attn_weights = F.softmax(scores.float(), dim=-1)
            attn_output = torch.matmul(attn_weights.to(v_cache.dtype), v_cache)
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output

    def tree_decoding(
            self,
            hidden_states,
            position_embeddings,
            cache_lens,
            K_Cache, # from LLM
            V_Cache, # from LLM
            llm_kv_len=None,
            tree_mask=None,
            ):

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=1)

        if K_Cache is not None:
            k_cache = K_Cache[:, :, :llm_kv_len, :].repeat_interleave(self.num_key_value_groups, dim=1).transpose(3, 2)
            v_cache = V_Cache[:, :, :llm_kv_len, :].repeat_interleave(self.num_key_value_groups, dim=1)
            scores = torch.matmul(query_states, k_cache) * self.softmax_scale
            attn_weights = F.softmax(scores.float(), dim=-1)
            attn_output = torch.matmul(attn_weights.to(v_cache.dtype), v_cache)
        
        else:
            _, current_kv_len, all_kv_len = tree_mask.size()
            range_indices = cache_lens.unsqueeze(-1) + self.range_indices[all_kv_len - current_kv_len : all_kv_len].unsqueeze(0)
            bsz_indices = self.range_indices[:bsz].unsqueeze(-1)
            self.K_Cache[bsz_indices, :, range_indices, :] = key_states.permute(0, 2, 1, 3)
            self.V_Cache[bsz_indices, :, range_indices, :] = value_states.permute(0, 2, 1, 3)

            all_cache_indices = cache_lens.unsqueeze(-1) + self.range_indices[:all_kv_len].unsqueeze(0)
            key_states = self.K_Cache[bsz_indices, :, all_cache_indices, :].permute(0, 2, 1, 3)
            value_states = self.V_Cache[bsz_indices, :, all_cache_indices, :].permute(0, 2, 1, 3)
            K_total = torch.cat((self.K_Cache[:, :, :cache_lens, :], key_states), dim=2)
            V_total = torch.cat((self.V_Cache[:, :, :cache_lens, :], value_states), dim=2)
            K_total = K_total.repeat_interleave(self.num_key_value_groups, dim=1).transpose(3, 2)
            V_total = V_total.repeat_interleave(self.num_key_value_groups, dim=1)
            total_mask = torch.cat((torch.zeros((bsz, current_kv_len, cache_lens), 
                                                device=query_states.device, dtype=bool), tree_mask == 0), dim=2)
            scores = torch.matmul(query_states, K_total) * self.softmax_scale
            scores = scores.masked_fill(total_mask, float('-inf'))
            attn_weights = F.softmax(scores.float(), dim=-1)
            attn_output = torch.matmul(attn_weights.to(V_total.dtype), V_total)
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size).half()
        attn_output = self.o_proj(attn_output)

        return attn_output


class Qwen2GlideDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.last_layer = (config.num_hidden_layers == self.layer_idx + 1)
        self.self_attn = GlideAttention(config, layer_idx)
        self.cross_attn = GlideAttention(config, layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_self_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_cross_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.config = config
    
    def set_max_gen_len(self, max_gen_len):
        self.self_attn.max_len = max_gen_len
        self.self_attn.range_indices = torch.arange(max_gen_len)
    
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        hidden_states,
        position_embeddings,
        llm_kv,
        cache_lens=None,
        exec_type=None,
        llm_kv_len=None,
        tree_mask=None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            cache_lens=cache_lens,
            exec_type="sa_" + exec_type,
            tree_mask=tree_mask,
        )
        hidden_states = residual + hidden_states

        # cross attn
        residual = hidden_states
        hidden_states = self.post_self_attention_layernorm(hidden_states)
        hidden_states = self.cross_attn(
            hidden_states=hidden_states, 
            position_embeddings=position_embeddings, 
            cache_lens=cache_lens, 
            exec_type="ca_" +exec_type, 
            k_cache=llm_kv[0], 
            v_cache=llm_kv[1], 
            llm_kv_len=llm_kv_len,
            tree_mask=tree_mask,
        )
        hidden_states += residual

        # ffn
        residual = hidden_states
        hidden_states = self.post_cross_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states += residual

        return hidden_states


class Qwen2Glide(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.glide = Qwen2GlideDecoderLayer(config, layer_idx=0)
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.lm_head.parameters():
            param.requires_grad = False

        for param in self.glide.parameters():
            param.requires_grad = True
        self.post_init()
    
    def compute_loss(self, hidden_states, labels):
        logits = self.lm_head(hidden_states).float()
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))
        return loss

    def forward(
        self,
        input_ids,
        labels,
        position_ids=None,
        cache_lens=None,
        **kwargs,
    ):
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.size(1))[None, :].to(input_ids.device)
        if cache_lens is not None:
            position_ids = position_ids + cache_lens

        labels[labels.eq(self.config.pad_token_id)] = -100
        with torch.inference_mode():
            llm_outputs = self.model(
                input_ids=input_ids,
                exec_type="glide_training",
                position_ids=position_ids,
                inputs_embeds=None,
                cache_lens=cache_lens,
                flex_attn=None,
            )
            llm_loss = self.compute_loss(llm_outputs.last_hidden_state, labels)
        position_ids = torch.arange(0, input_ids.size(1))[None, :].to(input_ids.device)
        llm_last_kv = llm_outputs.past_key_values
        position_embeddings = self.model.rotary_emb(llm_last_kv[0], position_ids)
        hidden_states = self.model.embed_tokens(input_ids)
        hidden_states = self.glide(hidden_states=hidden_states, position_embeddings=position_embeddings, llm_kv=llm_last_kv, exec_type="training")
        loss = self.compute_loss(hidden_states, labels)

        return CausalLMOutputWithPast(
            llm_loss=llm_loss,
            loss=loss,
        )
    
    def vanilla_generate(self, input_ids, max_gen_len=64, eos_id=151645):
        assert input_ids != None, "please give the input"
        bsz = input_ids.size(0)
        output_ids = input_ids.new_zeros((bsz, max_gen_len))
        
        self.set_max_gen_len(max_gen_len)
        
        cache_lens = input_ids.new_zeros((bsz)).int()
        hidden_states = self.model.forward(input_ids, exec_type="prefill").last_hidden_state
        input_len = input_ids.ne(self.config.pad_token_id).sum(dim=-1)
        output_ids[:, 0] = self.lm_head(hidden_states[range(bsz), input_len-1, :]).argmax(dim=-1)
        cache_lens += input_len

        # autoregressive decoding
        for _ in range(1, max_gen_len):
            input_ids = output_ids[range(bsz), cache_lens - input_len].view(bsz, -1)
            hidden_states = self.model.forward(input_ids, cache_lens=cache_lens.clone(), exec_type="decoding").last_hidden_state
            llm_output = self.lm_head(hidden_states[:, -1, :]).argmax(dim=-1)
            cache_lens += 1
            output_ids[range(bsz), cache_lens - input_len] = llm_output.view(-1)
            if (output_ids.eq(eos_id)).any():
                break

        return output_ids

    def spec_generate(self, input_ids, max_gen_len=64, eos_id=151645, gamma=4):
        assert input_ids != None, "please give the input"
        bsz = input_ids.size(0)
        output_ids = input_ids.new_zeros((bsz, max_gen_len + gamma + 2))
        spec_mask = input_ids.new_zeros((bsz, max_gen_len + gamma + 2))
        
        self.set_max_gen_len(max_gen_len + 128)
        self.glide.set_max_gen_len(max_gen_len + 128)
        
        cache_lens = input_ids.new_zeros((bsz)).int()
        hidden_states = self.model.forward(input_ids, exec_type="prefill")["last_hidden_state"]
        input_len = input_ids.ne(self.config.pad_token_id).sum(dim=-1)
        output_ids[:, 0] = self.lm_head(hidden_states[range(bsz), input_len-1, :]).argmax(dim=-1)
        spec_mask[:, 0] = 0
        cache_lens += input_len
        draft_cache_lens = cache_lens.clone()

        # Glide prefill
        hidden_states = self.model.embed_tokens(input_ids)
        position_ids = torch.arange(0, input_ids.size(1))[None, :].to(input_ids.device)
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)
        self.glide(
            hidden_states=hidden_states, 
            position_embeddings=position_embeddings, 
            llm_kv=(self.model.layers[-1].self_attn.K_Cache, self.model.layers[-1].self_attn.V_Cache), 
            cache_lens=draft_cache_lens.clone(), 
            llm_kv_len=cache_lens.clone(),
            exec_type="prefill",
        )

        # spec tokens
        double_flag = False
        spec_buffer = output_ids.new_zeros((bsz, gamma + 1))
        spec_buffer[:, 0] = output_ids[range(bsz), cache_lens - input_len]
        next_spec_start_token = output_ids.new_zeros((bsz, 2))
        next_spec_start_token[:, 0] = output_ids[:, 0]

        # autoregressive decoding
        for _ in range(1, max_gen_len):
            # speculative decoding
            for spec_steps in range(0, gamma):
                if spec_steps == 0:
                    if double_flag:
                        hidden_states = self.model.embed_tokens(next_spec_start_token[:, 0:2])
                        position_ids = torch.arange(0, 2)[None, :].to(input_ids.device) + draft_cache_lens[:, None]
                        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)
                    else:
                        hidden_states = self.model.embed_tokens(next_spec_start_token[:, 0, None])
                        position_embeddings = self.model.rotary_emb(hidden_states, draft_cache_lens[:, None])            

                else:
                    hidden_states = self.model.embed_tokens(spec_buffer[:, spec_steps, None])
                    position_embeddings = self.model.rotary_emb(hidden_states, draft_cache_lens[:, None])

                hidden_states = self.glide(hidden_states=hidden_states, position_embeddings=position_embeddings, 
                                           llm_kv=(self.model.layers[-1].self_attn.K_Cache, self.model.layers[-1].self_attn.V_Cache), 
                                           cache_lens=draft_cache_lens.clone(), llm_kv_len=cache_lens.clone(), exec_type="decoding")
                
                if double_flag and (spec_steps == 0):
                    # double batch id, if double accept, then gather the -1 token, else, gather the -2 token.
                    draft_cache_lens += 1 + double_input
                    current_logp = self.lm_head(hidden_states[:, -2:, :])
                    spec_buffer[:, spec_steps + 1] = current_logp.argmax(dim=-1)[range(bsz), double_input]
                else:
                    draft_cache_lens += 1
                    current_logp = self.lm_head(hidden_states[:, -1, :])
                    spec_buffer[:, spec_steps + 1] = current_logp.argmax(dim=-1).view(-1,)

            hidden_states = self.model.forward(spec_buffer, cache_lens=cache_lens.clone(), exec_type="decoding").last_hidden_state
            llm_verify_output = self.lm_head(hidden_states[:, -gamma - 1:, :]).argmax(dim=-1)
            verification = llm_verify_output[:, :-1].eq(spec_buffer[:, 1:]).cumprod(dim=-1)
            correct_len = verification.sum(dim=-1) + 1 # bonus token
            llm_verify_output[:, 1:] = llm_verify_output[:, 1:] * verification

            row_indices = torch.arange(bsz, device=cache_lens.device).unsqueeze(1)
            col_indices = (cache_lens - input_len).unsqueeze(1) + torch.arange(1, gamma + 1, device=cache_lens.device)
            output_ids[row_indices, col_indices] = llm_verify_output[:, :gamma]

            bonus_token = llm_verify_output[range(bsz), correct_len - 1]
            output_ids[range(bsz), cache_lens - input_len + correct_len] = bonus_token
            for i in range(bsz):
                spec_mask[i, cache_lens[i] - input_len[i] + 1 : cache_lens[i] - input_len[i] + correct_len[i]] = 1
            cache_lens += correct_len
            double_input = correct_len.eq(gamma + 1).to(torch.int)
            double_flag = double_input.eq(1).any()

            if double_flag:
                # vectorilize of the following code:
                # for i in range(bsz):
                #     if double_input[i] == 1:
                #         next_spec_start_token[i, :2] = llm_verify_output[i, -2:]
                #     else:
                #         next_spec_start_token[i, 0] = llm_verify_output[i, correct_len[i] - 1]
                next_spec_start_token[:, 0] = llm_verify_output[range(bsz), correct_len - 2]
                next_spec_start_token[:, 1] = llm_verify_output[range(bsz), correct_len - 1]
                next_spec_start_token[:, 0] = (1 - double_input.int()) * next_spec_start_token[:, 1] + double_input.int() * next_spec_start_token[:, 0]   
            else:
                next_spec_start_token[:, 0] = bonus_token

            spec_buffer[:, 0] = bonus_token            
            correct_len = correct_len.clamp(max=gamma)
            draft_cache_lens = cache_lens - double_input

            if (cache_lens - input_len).max() + gamma + 2 > output_ids.size(1):
                break
            if (output_ids.eq(eos_id)).any():
                break

        return output_ids[:, :max_gen_len]
