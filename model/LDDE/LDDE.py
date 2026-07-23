import torch
import torch.nn.functional as F
from torch import nn
from ..base import BaseConfig, ModelOutputs
from .LDDELayers import *

from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, TaskType, get_peft_model

import pdb


class LDDEConfig(BaseConfig):
    def __init__(self,
                 node_size,
                 num_classes):
        super(LDDEConfig, self).__init__(node_size=node_size,
                                         num_classes=num_classes)


class LDDE(nn.Module):
    """BrainNetCNN: Convolutional neural networks for brain networks; towards predicting neurodevelopment"""
    def __init__(self, config: LDDEConfig):
        super().__init__()
        self.config = config
        
        self.bnc1 = BrainNetCNN(config.node_size)
        self.fa = FeatureAlign()
        self.llm = AutoModel.from_pretrained("./model/chatglm-6b", trust_remote_code=True).float().cuda()
        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            task_type=TaskType.CAUSAL_LM,
            target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
            lora_dropout=0.1,
            bias="none",
        )
        self.llm = get_peft_model(self.llm, peft_config)
        self.fai = FeatureAlignInverse()
        self.gre = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 15)
        )
        
        self.bnc2 = BrainNetCNN(config.node_size)
        self.cla = nn.Linear(256, self.config.num_classes)

        self.loss_cla = nn.CrossEntropyLoss()
        self.loss_gre = nn.MSELoss()

    def forward(self, time_series, SFC, DFC, gender, age, education, labels, m_label):
        # pdb.set_trace()
        B, L, C, _ = DFC.shape
        DFC = DFC.reshape(-1, C, C)
        
        inputs_embeds = self.bnc1(DFC.unsqueeze(1))
        inputs_embeds = self.fa(inputs_embeds).reshape(B, L, -1).transpose(0, 1)
        hidden_state = self.llm.forward_from_embeds(inputs_embeds)
        hidden_state = hidden_state.sum(dim=1)#batch_size, d_model
        llm_hidden = self.fai(hidden_state)
        gre_logits = self.gre(hidden_state)
        
        sfeature = self.bnc2(SFC.unsqueeze(1))
        sfeature = sfeature + llm_hidden
        cla_logits = self.cla(sfeature)
        
        cla_loss = self.loss_cla(cla_logits, labels)
        gre_loss = self.loss_gre(gre_logits, m_label)
        
        # loss = cla_loss + gre_loss
        
        # pdb.set_trace()
        
        # if torch.isnan(loss).any() or torch.isinf(loss).any():
        #     pdb.set_trace()

        return ModelOutputs(logits=(cla_logits, gre_logits), loss=(cla_loss, gre_loss))
