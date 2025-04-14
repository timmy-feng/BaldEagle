import torch
import torch.nn as nn

from typing import List, Optional, Union

from transformers import PreTrainedModel
from transformers.models.llama.modeling_llama import LlamaModel

from transformers.cache_utils import Cache

class LlamaForCausalLMEagle(PreTrainedModel):
    def __init__(self, config):
        """
        Initializes the model with the Hugging Face structure:
        
          LlamaForCausalLM(
            (model): LlamaModel(...)
            (draft_fc): Linear(2*hidden_size, hidden_size, bias=False)  # for speculative decoding
          )
        """
        super().__init__(config)
        self.gradient_checkpointing = True
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size

        config.attn_implementation="flash_attention_2"

        # Monkey patch post init to no op - Hugging Face weight initialization makes model worse
        def noop_post_init(self):
            print("Running no op post init")
            pass
        LlamaModel.post_init = noop_post_init

        self.model = LlamaModel(config)
        
        # Follow EAGLE removal of norm layers
        del self.model.norm
        setattr(self.model, "norm", lambda x: x)
        del self.model.layers[0].input_layernorm
        setattr(self.model.layers[0], "input_layernorm", lambda x: x)
        
        # This projection layer maps the concatenated [token_emb; full_hidden] (of size 2*hidden_size)
        # down to hidden_size.
        self.model.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=True)
    
    def load_embedding_weights(self, weights):
        self.model.embed_tokens.weight = nn.Parameter(weights)

    def forward(
            self,
            hidden_state: torch.Tensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        token_emb = self.model.embed_tokens(input_ids)
        concat = torch.cat([token_emb, hidden_state], dim=-1)
        
        proj = self.model.fc(concat)

        outputs = self.model(
            input_ids=None,
            inputs_embeds=proj,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        return hidden_states
