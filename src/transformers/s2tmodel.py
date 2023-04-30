import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Speech2TextModel, Speech2TextForConditionalGeneration
import pdb


class CustomS2T(nn.Module):
    def __init__(self,config=None):
        super().__init__()
        stolen_config = Speech2TextModel.from_pretrained("facebook/s2t-small-librispeech-asr").config
        self.transformer = Speech2TextModel(stolen_config)
        self.lm_head = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr").lm_head
        self.extra_encoder_head = Speech2TextAttention(  256, 1)
        self.n_keywords = 10
        self.keyword_keys = nn.Parameter(torch.rand(self.n_keywords, 256))
        self.keyword_head = nn.Sequential(nn.Linear(self.n_keywords * 256, 512 ), nn.ReLU(),
                                          nn.Linear(512,256), nn.ReLU(),
                                          nn.Linear(256, self.n_keywords ))
        self.keyword_mode = False
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
        if self.keyword_mode:
            key_detect = self.extra_encoder_head(self.keyword_keys.reshape(1,10,256).repeat(24,1,1) , key_value_states=y.encoder_last_hidden_state)
            key_detect = key_detect[0]
            key_detect = self.keyword_head(key_detect.reshape(key_detect.shape[0], -1))
        
        y['cls_logit_out'] = cls_logit_out
        return y