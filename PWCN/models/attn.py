# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.dynamic_rnn import DynamicLSTM


class AOA(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(AOA, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.txt_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.asp_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * opt.hidden_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_indices, aspect_indices = inputs
        txt_len = torch.sum(text_indices != 0, dim=1)
        asp_len = torch.sum(aspect_indices != 0, dim=1)
        txt = self.embed(text_indices) 
        asp = self.embed(aspect_indices)
        txt_out, (_, _) = self.txt_lstm(txt, txt_len) 
        asp_out, (_, _) = self.asp_lstm(asp, asp_len)
        interaction_mat = torch.matmul(txt_out, torch.transpose(asp_out, 1, 2)) 
        alpha = F.softmax(interaction_mat, dim=1) # col-wise
        beta = F.softmax(interaction_mat, dim=2) # row-wise
        beta_avg = beta.mean(dim=1, keepdim=True) 
        gamma = torch.matmul(alpha, beta_avg.transpose(1, 2)) 
        weighted_sum = torch.matmul(torch.transpose(txt_out, 1, 2), gamma).squeeze(-1) 
        out = self.fc(weighted_sum)

        return out

class BiLSTMAttn(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(BiLSTMAttn, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.txt_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.asp_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * opt.hidden_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_indices, aspect_indices = inputs
        txt_len = torch.sum(text_indices != 0, dim=1)
        asp_len = torch.sum(aspect_indices != 0, dim=1)
        txt = self.embed(text_indices) 
        asp = self.embed(aspect_indices) 
        txt_out, (_, _) = self.txt_lstm(txt, txt_len) 
        asp_out, (_, _) = self.asp_lstm(asp, asp_len) 
        alpha_mat = torch.matmul(txt_out, torch.transpose(asp_out, 1, 2)) 
        alpha = F.softmax(alpha_mat.sum(2, keepdim=True), dim=1) 
        weighted_sum = torch.matmul(torch.transpose(txt_out, 1, 2), alpha).squeeze(-1) 
        out = self.fc(weighted_sum) 

        return out