import torch
import torch.nn as nn

from typing import Optional, Union, Tuple, Any

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
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Any,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # for EAGLE 3:
        # hidden_state is [token_emb; proj_emb]
        # only use proj_emb as the residual
        token_emb, residual = hidden_states[..., :self.hidden_size], hidden_states[..., self.hidden_size:]
        hidden_states = torch.cat([self.input_layernorm(token_emb), self.hidden_norm(residual)], dim=-1)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs

class LlamaForCausalLMEagle3(LlamaModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        # Monkey patch post init to no op - Hugging Face weight initialization makes model worse
        def noop_post_init():
            print("Running no op post init")
            pass

        self.post_init = noop_post_init
        super().__init__(config)

        config._attn_implementation = "flash_attention_2"

        self.layers[0] = LlamaDecoderLayerEagle3(config=config, layer_idx=0)

        if hasattr(config, "target_hidden_size"):
            hidden_size_in = config.target_hidden_size
        else:
            hidden_size_in = config.hidden_size

        # map the draft vocab to the target vocab
        self.register_buffer("d2t", torch.zeros(config.draft_vocab_size, dtype=torch.long))
        self.register_buffer("t2d", torch.zeros(config.vocab_size, dtype=torch.bool))

        # This projection layer maps the concatenated [low_hidden; mid_hidden; high_hidden] (of size 3*hidden_size)
        # down to hidden_size.
        self.fc = nn.Linear(3 * hidden_size_in, config.hidden_size, bias=False)

        # save the pre-norm hidden states for auxiliary decode
        self.pre_norm_hidden_states = None

        def save_pre_norm_hidden_states(module, input, output):
            self.pre_norm_hidden_states = input[0]

        self._pre_norm_hook = self.layers[0].hidden_norm.register_forward_hook(save_pre_norm_hidden_states)

        self.lm_head = nn.Linear(config.hidden_size, config.draft_vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def load_embedding_weights(self, weights):
        self.embed_tokens.weight = nn.Parameter(weights)
        # TODO: change behavior in case of tie_word_embeddings=False
        self.lm_head.weight = nn.Parameter(weights)

    def forward(
        self,
        hidden_state: torch.Tensor,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        token_emb = self.embed_tokens(input_ids)
        if hidden_state.shape[-1] != token_emb.shape[-1]:
            hidden_state = self.fc(hidden_state)
        concat = torch.cat([token_emb, hidden_state], dim=-1)

        outputs = super().forward(
            input_ids=None,
            inputs_embeds=concat,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]

        logits = self.lm_head(hidden_states)

        return Eagle3Output(
            hidden_states=hidden_states,
            pre_norm_hidden_states=self.pre_norm_hidden_states,
            logits=logits
        )
