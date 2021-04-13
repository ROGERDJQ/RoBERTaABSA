# -*- coding: utf-8 -*-

from layers.dynamic_rnn import DynamicLSTM
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
    
class PositionProximity(nn.Module):
    def __init__(self, opt):
        self.opt = opt
        super(PositionProximity, self).__init__()

    def forward(self, x, aspect_double_idx, text_len, aspect_len):
        batch_size, seq_len = x.shape[0], x.shape[1]
        weight = self.weight_matrix(aspect_double_idx, text_len, aspect_len, batch_size, seq_len).to(self.opt.device)
        x = weight.unsqueeze(2)*x
        return x

    def weight_matrix(self, aspect_double_idx, text_len, aspect_len, batch_size, seq_len):
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_double_idx[i,0]):
                weight[i].append(1-(aspect_double_idx[i,0]-j)/context_len)
            for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
                weight[i].append(0)
            for j in range(aspect_double_idx[i,1]+1, text_len[i]):
                weight[i].append(1-(j-aspect_double_idx[i,1])/context_len)
            for j in range(text_len[i], seq_len):
                weight[i].append(0)
        return torch.tensor(weight)

class PWCN_POS(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(PWCN_POS, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.embed_dropout = nn.Dropout(0.3)
        self.txt_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.proximity = PositionProximity(opt)
        self.convs3 = nn.Conv1d(2*opt.hidden_dim, 2*opt.hidden_dim, 3, padding=1)   
        self.fc = nn.Linear(2*opt.hidden_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_indices, aspect_indices, left_indices = inputs
        txt_len = torch.sum(text_indices != 0, dim=1)
        asp_len = torch.sum(aspect_indices != 0, dim=1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(-1), (left_len+asp_len-1).unsqueeze(-1)], dim=-1)
        txt_out = self.embed_dropout(self.embed(text_indices))
        txt_out, (_, _) = self.txt_lstm(txt_out, txt_len)
        z = F.relu(self.convs3(
            self.proximity(txt_out, aspect_double_idx, txt_len, asp_len).transpose(1, 2)))  # [(N,Co,L), ...]*len(Ks)
        z = F.max_pool1d(z, z.size(2)).squeeze(2)
        out = self.fc(z)
        return out
