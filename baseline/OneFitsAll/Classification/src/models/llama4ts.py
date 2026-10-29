"""OFA classification variant with the project's frozen 8B Llama backbone.

Adaptation of gpt4ts.py:
  - GPT-2 -> AutoModel (Llama) loaded from config['llm_path'] in bf16
  - first `gpt_layers` (6) LLM layers only, all frozen except layer norms
    (Llama has no wpe; RoPE position info is not trainable)
  - readout: mean-pool over patches -> LayerNorm -> Linear head
    (the original flatten head would be ~6M params at d_model=4096)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from einops import rearrange

from models.embed import DataEmbedding


class llama4ts(nn.Module):

    def __init__(self, config, data):
        super(llama4ts, self).__init__()
        self.pred_len = 0
        self.seq_len = data.max_seq_len
        self.max_len = data.max_seq_len
        self.patch_size = config['patch_size']
        self.stride = config['stride']
        self.gpt_layers = 6
        self.feat_dim = data.feature_df.shape[1]
        self.num_classes = len(data.class_names)
        self.d_model = config['d_model']

        self.patch_num = (self.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1
        self.enc_embedding = DataEmbedding(self.feat_dim * self.patch_size,
                                           config['d_model'],
                                           config['dropout'])

        self.llm = AutoModel.from_pretrained(
            config['llm_path'], torch_dtype=torch.bfloat16)
        # Keep only the first gpt_layers transformer blocks (paper design:
        # shallow frozen stack; slicing a ModuleList unregisters the rest).
        self.llm.layers = nn.ModuleList(self.llm.layers[:self.gpt_layers])
        for name, param in self.llm.named_parameters():
            # GPT-2 version trains ln/wpe; Llama equivalent is its RMSNorms.
            param.requires_grad = ('layernorm' in name)

        self.act = F.gelu
        self.dropout = nn.Dropout(0.1)
        self.ln_proj = nn.LayerNorm(config['d_model'])
        self.out_layer = nn.Linear(config['d_model'], self.num_classes)

    def forward(self, x_enc, x_mark_enc):
        B, L, M = x_enc.shape

        input_x = rearrange(x_enc, 'b l m -> b m l')
        input_x = self.padding_patch_layer(input_x)
        input_x = input_x.unfold(dimension=-1, size=self.patch_size,
                                 step=self.stride)
        input_x = rearrange(input_x, 'b m n p -> b n (p m)')

        outputs = self.enc_embedding(input_x, None)

        # Llama inputs_embeds must match the model weight dtype (bf16)
        outputs = self.llm(
            inputs_embeds=outputs.to(torch.bfloat16)).last_hidden_state
        outputs = outputs.float()

        outputs = self.dropout(self.act(outputs))
        outputs = outputs.mean(dim=1)          # mean-pool over patches
        outputs = self.ln_proj(outputs)
        outputs = self.out_layer(outputs)

        return outputs
