import torch
import torch.nn as nn

from typing import Optional, Tuple

from transformers.models.llama.modeling_llama import LlamaModel, LlamaDecoderLayer, LlamaAttention, LlamaConfig

from transformers.cache_utils import Cache
from transformers.utils import ModelOutput

from dataclasses import dataclass

@dataclass
class Eagle3Output(ModelOutput):
    logits: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    pre_norm_hidden_states: Optional[torch.Tensor] = None


class LlamaAttentionEagle3(LlamaAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        # override the QKV projection input dimension to 2 * hidden_size
        self.q_proj = nn.Linear(
            2 * config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            2 * config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            2 * config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )

class LlamaDecoderLayerEagle3(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = LlamaAttentionEagle3(config=config, layer_idx=layer_idx)
        self.hidden_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = torch.cat([
            self.input_layernorm(inputs_embeds),
            self.hidden_norm(hidden_states)
        ], dim=-1)

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class LlamaForCausalLMEagle3(LlamaModel):
    def __init__(self, config):
        # Monkey patch post init to no op - Hugging Face weight initialization makes model worse
        def noop_post_init():
            print("Running no op post init")
            pass

        self.post_init = noop_post_init
        super().__init__(config)

        config._attn_implementation = "flash_attention_2"

        # rename layers to midlayer for EAGLE 3 compatibility
        del self.layers
        self.midlayer = LlamaDecoderLayerEagle3(config=config, layer_idx=0)

        # map the draft vocab to the target vocab
        self.register_buffer("d2t", torch.zeros(config.draft_vocab_size, dtype=torch.long))
        self.register_buffer("t2d", torch.zeros(config.vocab_size, dtype=torch.bool))

        # This projection layer maps the concatenated [low_hidden; mid_hidden; high_hidden]
        # down to hidden_size.
        self.fc = nn.Linear(3 * config.hidden_size, config.hidden_size, bias=False)

        self.lm_head = nn.Linear(config.hidden_size, config.draft_vocab_size, bias=False)

    def state_dict(self, *args, **kwargs):
        # Prevent saving the embedding weights
        state_dict = super().state_dict(*args, **kwargs)
        del state_dict['embed_tokens.weight']
        return state_dict

    def load_embedding_weights(self, weights):
        self.embed_tokens.weight = nn.Parameter(weights)

    def forward(
        self,
        hidden_state: torch.Tensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> Eagle3Output:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if hidden_state.shape[-1] != inputs_embeds.shape[-1]:
            hidden_state = self.fc(hidden_state)

        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        pre_norm_hidden_states = self.midlayer(
            inputs_embeds=inputs_embeds,
            hidden_states=hidden_state,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )

        hidden_states = self.norm(pre_norm_hidden_states)

        logits = self.lm_head(hidden_states)

        return Eagle3Output(
            hidden_states=hidden_states,
            pre_norm_hidden_states=pre_norm_hidden_states,
            logits=logits
        )
