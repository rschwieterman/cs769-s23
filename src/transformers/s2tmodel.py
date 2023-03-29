import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Speech2TextModel, Speech2TextForConditionalGeneration
import pdb


class customs2t(nn.Module):
    def __init__(self,config=None):
        super().__init__()
        self.transformer = Speech2TextModel.from_pretrained("facebook/s2t-small-librispeech-asr")
        self.lm_head = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr").lm_head
        print("done")

    def train(self):
        self.transformer.encoder.train()
        self.transformer.decoder.train()
        self.lm_head.train()
    def forward(self, x,attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,):
        #print("forward")
        y = self.transformer(input_features=x, decoder_input_ids = decoder_input_ids, decoder_attention_mask = decoder_attention_mask)
        cls_logit_out = self.lm_head(y.last_hidden_state)
        y['cls_logit_out'] = cls_logit_out
        return y