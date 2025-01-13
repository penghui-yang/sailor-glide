import math
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F


from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2MLP,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
    apply_rotary_pos_emb,
)


logger = logging.get_logger(__name__)


_CHECKPOINT_FOR_DOC = "Qwen/Qwen2-7B-beta"
_CONFIG_FOR_DOC = "Qwen2Config"


class Qwen2Attention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: Qwen2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

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
        self.max_len = 512
        self.log_ratio = math.log(0.7)
        self.prefix_lens = None
        self.layer_idx = layer_idx
        self.last_layer = (config.num_hidden_layers == self.layer_idx + 1)
        self.softmax_scale = 1 / (128 ** 0.5)
        self.range_indices = None


    def forward(
        self,
        hidden_states,
        position_embeddings,
        cache_lens=None,
        flex_attn=None,
        tree_mask=None,
        exec_type="training",
        induction_head=False,
    ):
        
        kv_cache = None
        if exec_type == "prefill":
            y = self.prefill(hidden_states, position_embeddings)
        elif exec_type == "decoding":
            y = self.decoding(hidden_states, position_embeddings, cache_lens)
        elif exec_type == "tree_decoding":
            y = self.tree_decoding(hidden_states, position_embeddings, cache_lens, tree_mask)
        else:
            raise ValueError(f"Unknown inference_type: {exec_type}")
        return y, kv_cache
    
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
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            if self.last_layer:
                scores = torch.matmul(query_states * self.softmax_scale, key_states)
            else:
                scores = torch.matmul(query_states, key_states) * self.softmax_scale
        scores = scores.masked_fill(mask, float('-inf'))
        attn_weights = F.softmax(scores.float(), dim=-1)
        attn_output = torch.matmul(attn_weights.to(value_states.dtype), value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        self.range_indices = torch.arange(1024, device=self.K_Cache.device)
        attn_output = self.o_proj(attn_output)

        return attn_output

    def decoding(
            self,
            hidden_states,
            position_embeddings,
            cache_lens,
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

        self.K_Cache[:, :, cache_lens : cache_lens + q_len, :] = key_states
        self.V_Cache[:, :, cache_lens : cache_lens + q_len, :] = value_states
        K_total = torch.cat((self.K_Cache[:, :, :cache_lens, :], key_states), dim=2)
        V_total = torch.cat((self.V_Cache[:, :, :cache_lens, :], value_states), dim=2)
        K_total = K_total.repeat_interleave(self.num_key_value_groups, dim=1).transpose(3, 2)
        V_total = V_total.repeat_interleave(self.num_key_value_groups, dim=1)
        mask = torch.triu(torch.ones(q_len, q_len, device=query_states.device), diagonal=1).bool()
        total_mask = torch.cat((torch.zeros((q_len, cache_lens), device=query_states.device, dtype=bool), mask), dim=1)
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            if self.last_layer:
                scores = torch.matmul(query_states * self.softmax_scale, K_total)
            else:
                scores = torch.matmul(query_states, K_total) * self.softmax_scale
        scores = scores.masked_fill(total_mask, float('-inf'))
        attn_weights = F.softmax(scores.float(), dim=-1)
        attn_output = torch.matmul(attn_weights.to(V_total.dtype), V_total)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output

    def tree_decoding(
            self,
            hidden_states,
            position_embeddings,
            cache_lens,
            tree_mask=None,
            ):
        '''
        tree_mask: bsz fseq fseq (flatten_seqlen)
        '''
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=1)

        # self.batch_indices # torch.arange(1024, device=K_cache.device)

        if tree_mask is None:
            assert q_len == 1, "You are in the first step of tree decoding, thus you should not input qlen > 2 without tree mask"
            self.K_Cache[:, :, cache_lens : cache_lens + q_len, :] = key_states
            self.V_Cache[:, :, cache_lens : cache_lens + q_len, :] = value_states
            K_total = torch.cat((self.K_Cache[:, :, :cache_lens, :], key_states), dim=2)
            V_total = torch.cat((self.V_Cache[:, :, :cache_lens, :], value_states), dim=2)
            K_total = K_total.repeat_interleave(self.num_key_value_groups, dim=1).transpose(3, 2)
            V_total = V_total.repeat_interleave(self.num_key_value_groups, dim=1)
            mask = torch.triu(torch.ones(q_len, q_len, device=query_states.device), diagonal=1).bool()
            total_mask = torch.cat((torch.zeros((q_len, cache_lens), device=query_states.device, dtype=bool), mask), dim=1)

        else:
            _, current_kv_len, all_kv_len = tree_mask.size()
            range_indices = cache_lens.unsqueeze(-1) + self.range_indices[:all_kv_len].unsqueeze(0)  # 计算范围
            bsz_indices = self.range_indices[:bsz].unsqueeze(-1)
            self.K_Cache[bsz_indices, :, range_indices, :] = key_states.permute(0, 2, 1, 3)
            self.V_Cache[bsz_indices, :, range_indices, :] = value_states.permute(0, 2, 1, 3)
            K_total = torch.cat((self.K_Cache[:, :, :cache_lens, :], key_states), dim=2)
            V_total = torch.cat((self.V_Cache[:, :, :cache_lens, :], value_states), dim=2)
            K_total = K_total.repeat_interleave(self.num_key_value_groups, dim=1).transpose(3, 2)
            V_total = V_total.repeat_interleave(self.num_key_value_groups, dim=1)
            total_mask = torch.cat((torch.zeros((bsz, current_kv_len, cache_lens), 
                                                 device=query_states.device, dtype=bool), tree_mask == 0), dim=2)
        
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            if self.last_layer:
                scores = torch.matmul(query_states * self.softmax_scale, K_total)
            else:
                scores = torch.matmul(query_states, K_total) * self.softmax_scale
        scores = scores.masked_fill(total_mask, float('-inf'))
        attn_weights = F.softmax(scores.float(), dim=-1)
        attn_output = torch.matmul(attn_weights.to(V_total.dtype), V_total)
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size).to(hidden_states.dtype)
        attn_output = self.o_proj(attn_output)

        return attn_output


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.last_layer = (config.num_hidden_layers == self.layer_idx + 1)
        self.self_attn = Qwen2Attention(config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states,
        position_embeddings,  # will become mandatory in v4.46
        cache_lens=None,
        flex_attn=None,
        exec_type=None,
        tree_mask=None,
        induction_head=False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, kv_cache = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            cache_lens=cache_lens,
            flex_attn=flex_attn,
            exec_type=exec_type,
            tree_mask=tree_mask,
            induction_head=induction_head,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, kv_cache)

        return outputs

class Qwen2PreTrainedModel(PreTrainedModel):
    config_class = Qwen2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

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

class Qwen2Model(Qwen2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: Qwen2Config
    """

    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value
    

    def forward(
        self,
        input_ids,
        position_ids=None,
        inputs_embeds=None,
        cache_lens=None,
        flex_attn=None,
        exec_type=None,
        tree_mask=None,
        induction_head=False,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if position_ids is None:
            if tree_mask is None:
                position_ids = torch.arange(0, input_ids.size(1))[None, :].to(input_ids.device)
            else:
                position_ids = tree_mask.sum(dim=-1) - 1
            if cache_lens is not None:
                position_ids = position_ids + cache_lens[:, None]
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)


        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers:

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    position_embeddings,
                    cache_lens,
                    flex_attn,
                    exec_type,
                    tree_mask,
                    induction_head,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    position_embeddings,
                    cache_lens,
                    flex_attn,
                    exec_type,
                    tree_mask,
                    induction_head,
                )

            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)

        if exec_type == "glide_training":
            kv_cache = layer_outputs[1]
        else:
            kv_cache = None
        # add hidden states from the last decoder layer

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=kv_cache,
            hidden_states=None,
            attentions=None,
        )


class Qwen2ForCausalLM(Qwen2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.eod = 151645
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def set_max_gen_len(self, max_gen_len):
        for layer in self.model.layers:
            layer.self_attn.max_len = max_gen_len
    
    def set_log_ratio(self, log_ratio):
        for layer in self.model.layers:
            layer.self_attn.log_ratio = log_ratio
    
    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        cache_lens=None,
        exec_type="training",
        induction_head=False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        if exec_type == "free_training":

            bsz, seqlen = position_ids.size()
            eod_mask = position_ids.eq(self.eod)
            eod_indices = torch.nonzero(eod_mask, as_tuple=False)


            assert eod_indices.size(0) == bsz * self.sample_num, "dataset needs all batch samples have same output samples equasl to self.sample_num"

            eod_col = eod_indices[:, 1].view(bsz, 10)
            prefix_end, doc_end = eod_col[:, 0], eod_col[:, 1:]
            # block_mask = construct_doc_mask(bsz, prefix_end, doc_end, seqlen)
            # block_mask = create_block_mask(construct_doc_mask, B=None, H=None, Q_LEN=8192, KV_LEN=8192, _compile=True)
            # flex_attn = torch.compile(partial(flex_attention, block_mask=block_mask, enable_gqa=True))
            flex_attn = None
        else:
            flex_attn = None

        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            cache_lens=cache_lens,
            flex_attn=flex_attn,
            exec_type=exec_type,
            induction_head=induction_head,
        )

        hidden_states = outputs[0]
        last_kv = outputs[1]

        loss = None
        logits = self.lm_head(hidden_states[:, -128:, :]).float()

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=last_kv,
            hidden_states=None,
            attentions=None,
        )
