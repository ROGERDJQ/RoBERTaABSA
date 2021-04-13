import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, BertPreTrainedModel, BertTokenizer, RobertaTokenizer, RobertaModel, RobertaConfig
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from model_gcn import GAT, GCN, Rel_GAT
from model_utils import LinearAttention, DotprodAttention, RelationAttention, Highway, mask_logits
from tree import *
from fastNLP import seq_len_to_mask
from fastNLP.modules import LSTM


class Aspect_Text_GAT_ours(nn.Module):
    """
    Full model in reshaped tree
    """
    def __init__(self, args, dep_tag_num, pos_tag_num):
        super(Aspect_Text_GAT_ours, self).__init__()
        self.args = args

        num_embeddings, embed_dim = args.glove_embedding.shape
        self.embed = nn.Embedding(num_embeddings, embed_dim, padding_idx=0)
        self.embed.weight = nn.Parameter(
            args.glove_embedding, requires_grad=False)

        self.dropout = nn.Dropout(args.dropout)

        if args.highway:
            self.highway_dep = Highway(args.num_layers, args.embedding_dim)
            self.highway = Highway(args.num_layers, args.embedding_dim)
        if args.num_layers>1:
            self.bilstm = LSTM(input_size=args.embedding_dim, hidden_size=args.hidden_size,
                                  bidirectional=True, batch_first=True, num_layers=args.num_layers,
                                  dropout=0.5)
        else:
            self.bilstm = LSTM(input_size=args.embedding_dim, hidden_size=args.hidden_size,
                                  bidirectional=True, batch_first=True, num_layers=args.num_layers)
        gcn_input_dim = args.hidden_size * 2

        # if args.gat:
        self.gat_dep = [RelationAttention(in_dim = args.embedding_dim).to(args.device) for i in range(args.num_heads)]
        if args.gat_attention_type == 'linear':
            self.gat = [LinearAttention(in_dim = gcn_input_dim, mem_dim = gcn_input_dim).to(args.device) for i in range(args.num_heads)] # we prefer to keep the dimension unchanged
        elif args.gat_attention_type == 'dotprod':
            self.gat = [DotprodAttention() for i in range(args.num_heads)]
        else:
            # reshaped gcn
            self.gat = nn.Linear(gcn_input_dim, gcn_input_dim)

        self.dep_embed = nn.Embedding(dep_tag_num, args.embedding_dim, padding_idx=0)
        torch.nn.init.uniform_(self.dep_embed.weight, a=-1. / math.sqrt(args.embedding_dim),
                               b=1. / math.sqrt(args.embedding_dim))

        last_hidden_size = args.hidden_size * 4

        layers = [
            nn.Linear(last_hidden_size, args.final_hidden_size), nn.Dropout(0.5), nn.ReLU()]
        for _ in range(args.num_mlps-1):
            layers += [nn.Linear(args.final_hidden_size,
                                 args.final_hidden_size), nn.Dropout(0.5), nn.ReLU()]
        self.fcs = nn.Sequential(*layers)
        self.fc_final = nn.Linear(args.final_hidden_size, args.num_classes)
        self._reset_params()

    def _reset_params(self):
        for name, p in self.named_parameters():
            if 'embed' in name:
                continue
            if p.requires_grad:
                if len(p.shape) > 1:
                    torch.nn.init.xavier_uniform_(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def forward(self, sentence, aspect, pos_class, dep_tags, text_len, aspect_len, dep_rels, dep_heads, aspect_position, dep_dirs):
        '''
        Forward takes:
            sentence: sentence_id of size (batch_size, text_length)
            aspect: aspect_id of size (batch_size, aspect_length)
            pos_class: pos_tag_id of size (batch_size, text_length)
            dep_tags: dep_tag_id of size (batch_size, text_length)
            text_len: (batch_size,) length of each sentence
            aspect_len: (batch_size, ) aspect length of each sentence
            dep_rels: (batch_size, text_length) relation
            dep_heads: (batch_size, text_length) which node adjacent to that node
            aspect_position: (batch_size, text_length) mask, with the position of aspect as 1 and others as 0
            dep_dirs: (batch_size, text_length) the directions each node to the aspect
        '''
        fmask = seq_len_to_mask(text_len).float()
        # fmask = (torch.zeros_like(sentence) != sentence).float()  # (N，L), pad为0
        # dmask = (torch.zeros_like(dep_tags) != dep_tags).float()  # (N ,L)
        if self.training:
            mask = torch.rand(sentence.size()).lt(0.02).to(sentence.device)
            sentence = sentence.masked_fill(mask, 0)
            # mask = torch.rand(aspect.size()).lt(0.01).to(sentence.device)
            # aspect = aspect.masked_fill(mask, 0)

        feature = self.embed(sentence)  # (N, L, D)
        aspect_feature = self.embed(aspect) # (N, L', D)
        feature = self.dropout(feature)
        aspect_feature = self.dropout(aspect_feature)

        if self.args.highway:
            feature = self.highway(feature)
            aspect_feature = self.highway(aspect_feature)

        feature, _ = self.bilstm(feature, seq_len=text_len) # (N,L,D)
        aspect_feature, _ = self.bilstm(aspect_feature, seq_len=aspect_len) #(N,L,D)
        aspect_mask = seq_len_to_mask(aspect_len)

        # aspect_feature = aspect_feature.masked_fill(aspect_mask.eq(0).unsqueeze(-1), 0)
        # aspect_feature = aspect_feature.sum(dim=1)/aspect_len.unsqueeze(1).float()
        aspect_feature = aspect_feature.masked_fill(aspect_mask.eq(0).unsqueeze(-1), -10000)
        aspect_feature, _ = aspect_feature.max(dim = 1)
        # aspect_feature = aspect_feature.mean(dim=1)

        ############################################################################################
        # do gat thing
        dep_feature = self.dep_embed(dep_tags)
        # dep_feature = self.dropout(dep_feature)
        dep_feature = F.dropout(dep_feature, p=0.7, training=self.training)
        if self.args.highway:
            dep_feature = self.highway_dep(dep_feature)

        dep_out = [g(feature, dep_feature, fmask).unsqueeze(1) for g in self.gat_dep]  # (N, 1, D) * num_heads
        dep_out = torch.cat(dep_out, dim = 1) # (N, H, D)
        # dep_out = dep_out.mean(dim = 1) # (N, D)
        dep_out, _ = dep_out.max(dim = 1) # (N, D)

        if self.args.gat_attention_type == 'gcn':
            gat_out = self.gat(feature) # (N, L, D)
            fmask = fmask.unsqueeze(2)
            gat_out = gat_out * fmask
            gat_out = F.relu(torch.sum(gat_out, dim = 1)) # (N, D)
        else:
            gat_out = [g(feature, aspect_feature, fmask).unsqueeze(1) for g in self.gat]
            gat_out = torch.cat(gat_out, dim=1)
            # gat_out = gat_out.mean(dim=1)
            gat_out, _ = gat_out.max(dim=1)

        feature_out = torch.cat([dep_out,  gat_out], dim = 1) # (N, D')
        # feature_out = gat_out
        #############################################################################################
        feature_out = self.dropout(feature_out)
        # feature_out = F.dropout(feature_out, p=0.3, training=self.training)
        x = self.fcs(feature_out)
        logit = self.fc_final(x)
        return logit


class Aspect_Text_GAT_only(nn.Module):
    """
    reshape tree in GAT only
    """
    def __init__(self, args, dep_tag_num, pos_tag_num):
        super(Aspect_Text_GAT_only, self).__init__()
        self.args = args

        num_embeddings, embed_dim = args.glove_embedding.shape
        self.embed = nn.Embedding(num_embeddings, embed_dim)
        self.embed.weight = nn.Parameter(
            args.glove_embedding, requires_grad=False)

        self.dropout = nn.Dropout(args.dropout)
        self.tanh = nn.Tanh()

        if args.highway:
            self.highway = Highway(args.num_layers, args.embedding_dim)

        self.bilstm = nn.LSTM(input_size=args.embedding_dim, hidden_size=args.hidden_size,
                                  bidirectional=True, batch_first=True, num_layers=args.num_layers)
        gcn_input_dim = args.hidden_size * 2

        # if args.gat:
        if args.gat_attention_type == 'linear':
            self.gat = [LinearAttention(in_dim = gcn_input_dim, mem_dim = gcn_input_dim).to(args.device) for i in range(args.num_heads)] # we prefer to keep the dimension unchanged
        elif args.gat_attention_type == 'dotprod':
            self.gat = [DotprodAttention().to(args.device) for i in range(args.num_heads)]
        else:
            # reshaped gcn
            self.gat = nn.Linear(gcn_input_dim, gcn_input_dim)


        last_hidden_size = args.hidden_size * 2

        layers = [
            nn.Linear(last_hidden_size, args.final_hidden_size), nn.ReLU()]
        for _ in range(args.num_mlps-1):
            layers += [nn.Linear(args.final_hidden_size,
                                 args.final_hidden_size), nn.ReLU()]
        self.fcs = nn.Sequential(*layers)
        self.fc_final = nn.Linear(args.final_hidden_size, args.num_classes)

    def forward(self, sentence, aspect, pos_class, dep_tags, text_len, aspect_len, dep_rels, dep_heads, aspect_position, dep_dirs):
        '''
        Forward takes:
            sentence: sentence_id of size (batch_size, text_length)
            aspect: aspect_id of size (batch_size, aspect_length)
            pos_class: pos_tag_id of size (batch_size, text_length)
            dep_tags: dep_tag_id of size (batch_size, text_length)
            text_len: (batch_size,) length of each sentence
            aspect_len: (batch_size, ) aspect length of each sentence
            dep_rels: (batch_size, text_length) relation
            dep_heads: (batch_size, text_length) which node adjacent to that node
            aspect_position: (batch_size, text_length) mask, with the position of aspect as 1 and others as 0
            dep_dirs: (batch_size, text_length) the directions each node to the aspect
        '''
        fmask = (torch.zeros_like(sentence) != sentence).float()  # (N，L)
        dmask = (torch.zeros_like(dep_tags) != dep_tags).float()  # (N ,L)

        feature = self.embed(sentence)  # (N, L, D)
        aspect_feature = self.embed(aspect) # (N, L', D)
        feature = self.dropout(feature)
        aspect_feature = self.dropout(aspect_feature)


        if self.args.highway:
            feature = self.highway(feature)
            aspect_feature = self.highway(aspect_feature)

        feature, _ = self.bilstm(feature) # (N,L,D)
        aspect_feature, _ = self.bilstm(aspect_feature) #(N,L,D)

        aspect_feature = aspect_feature.mean(dim = 1) # (N, D)

        ############################################################################################

        if self.args.gat_attention_type == 'gcn':
            gat_out = self.gat(feature) # (N, L, D)
            fmask = fmask.unsqueeze(2)
            gat_out = gat_out * fmask
            gat_out = F.relu(torch.sum(gat_out, dim = 1)) # (N, D)

        else:
            gat_out = [g(feature, aspect_feature, fmask).unsqueeze(1) for g in self.gat]
            gat_out = torch.cat(gat_out, dim=1)
            gat_out = gat_out.mean(dim=1)

        feature_out = gat_out # (N, D')
        # feature_out = gat_out
        #############################################################################################
        x = self.dropout(feature_out)
        x = self.fcs(x)
        logit = self.fc_final(x)
        return logit


class Pure_Bert(nn.Module):
    '''
    Bert for sequence classification.
    '''

    def __init__(self, args, hidden_size=256):
        super(Pure_Bert, self).__init__()

        config = BertConfig.from_pretrained(args.bert_model_dir)
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model_dir)
        self.bert = BertModel.from_pretrained(
            args.bert_model_dir, config=config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        layers = [nn.Linear(
            config.hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, args.num_classes)]
        self.classifier = nn.Sequential(*layers)

    def forward(self, input_ids, token_type_ids):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids)
        # pool output is usually *not* a good summary of the semantic content of the input,
        # you're often better with averaging or poolin the sequence of hidden-states for the whole input sequence.
        pooled_output = outputs[1]
        # pooled_output = torch.mean(pooled_output, dim = 1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits


class Aspect_Bert_GAT(nn.Module):
    '''
    R-GAT with bert
    '''

    def __init__(self, args, dep_tag_num, pos_tag_num):
        super(Aspect_Bert_GAT, self).__init__()
        self.args = args

        # Bert
        config = BertConfig.from_pretrained(args.bert_model_dir)
        self.bert = BertModel.from_pretrained(
            args.bert_model_dir, config=config, from_tf =False)
        self.dropout_bert = nn.Dropout(config.hidden_dropout_prob)
        self.dropout = nn.Dropout(args.dropout)
        args.embedding_dim = config.hidden_size  # 768

        if args.highway:
            self.highway_dep = Highway(args.num_layers, args.embedding_dim)
            self.highway = Highway(args.num_layers, args.embedding_dim)


        gcn_input_dim = args.embedding_dim

        # GAT
        self.gat_dep = [RelationAttention(in_dim=args.embedding_dim).to(args.device) for i in range(args.num_heads)]


        self.dep_embed = nn.Embedding(dep_tag_num, args.embedding_dim)

        last_hidden_size = args.embedding_dim * 2
        layers = [
            nn.Linear(last_hidden_size, args.final_hidden_size), nn.ReLU()]
        for _ in range(args.num_mlps - 1):
            layers += [nn.Linear(args.final_hidden_size,
                                 args.final_hidden_size), nn.ReLU()]
        self.fcs = nn.Sequential(*layers)
        self.fc_final = nn.Linear(args.final_hidden_size, args.num_classes)

    def forward(self, input_ids, input_aspect_ids, word_indexer, aspect_indexer,input_cat_ids,segment_ids, pos_class, dep_tags, text_len, aspect_len, dep_rels, dep_heads, aspect_position, dep_dirs):
        fmask = (torch.ones_like(word_indexer) != word_indexer).float()  # (N，L) 为1的地方是pad
        fmask[:,0] = 1
        outputs = self.bert(input_cat_ids, token_type_ids = segment_ids)
        feature_output = outputs[0] # (N, L, D)
        pool_out = outputs[1] #(N, D)

        # index select, back to original batched size.
        feature = torch.stack([torch.index_select(f, 0, w_i)
                               for f, w_i in zip(feature_output, word_indexer)])

        ############################################################################################
        # do gat thing
        dep_feature = self.dep_embed(dep_tags)
        if self.args.highway:
            dep_feature = self.highway_dep(dep_feature)

        # 为1的地方信息保留
        dep_out = [g(feature, dep_feature, fmask).unsqueeze(1) for g in self.gat_dep]  # (N, 1, D) * num_heads
        dep_out = torch.cat(dep_out, dim=1)  # (N, H, D)
        dep_out = dep_out.mean(dim=1)  # (N, D)


        feature_out = torch.cat([dep_out,  pool_out], dim=1)  # (N, D')
        # feature_out = gat_out
        #############################################################################################
        x = self.dropout(feature_out)
        x = self.fcs(x)
        logit = self.fc_final(x)
        return logit


class Aspect_Roberta_GAT(nn.Module):
    '''
    R-GAT with bert
    '''

    def __init__(self, args, dep_tag_num, pos_tag_num):
        super(Aspect_Roberta_GAT, self).__init__()
        self.args = args

        # Bert
        config = RobertaConfig.from_pretrained(args.bert_model_dir)
        self.bert = RobertaModel.from_pretrained(
            args.bert_model_dir, config=config, from_tf =False)
        self.dropout_bert = nn.Dropout(config.hidden_dropout_prob)
        self.dropout = nn.Dropout(args.dropout)
        args.embedding_dim = config.hidden_size  # 768

        if args.highway:
            self.highway_dep = Highway(args.num_layers, args.embedding_dim)
            self.highway = Highway(args.num_layers, args.embedding_dim)


        gcn_input_dim = args.embedding_dim

        # GAT
        self.gat_dep = [RelationAttention(in_dim=args.embedding_dim).to(args.device) for i in range(args.num_heads)]


        self.dep_embed = nn.Embedding(dep_tag_num, args.embedding_dim)

        last_hidden_size = args.embedding_dim * 2
        layers = [
            nn.Linear(last_hidden_size, args.final_hidden_size), nn.ReLU()]
        for _ in range(args.num_mlps - 1):
            layers += [nn.Linear(args.final_hidden_size,
                                 args.final_hidden_size), nn.ReLU()]
        self.fcs = nn.Sequential(*layers)
        self.fc_final = nn.Linear(args.final_hidden_size, args.num_classes)

    def forward(self, input_ids, input_aspect_ids, word_indexer, aspect_indexer,input_cat_ids,segment_ids, pos_class, dep_tags, text_len, aspect_len, dep_rels, dep_heads, aspect_position, dep_dirs):
        fmask = (torch.ones_like(word_indexer) != word_indexer).float()  # (N，L)
        fmask[:,0] = 1
        outputs = self.bert(input_cat_ids, token_type_ids = segment_ids)
        feature_output = outputs[0] # (N, L, D)
        pool_out = outputs[1] #(N, D)

        # index select, back to original batched size.
        feature = torch.stack([torch.index_select(f, 0, w_i)
                               for f, w_i in zip(feature_output, word_indexer)])

        ############################################################################################
        # do gat thing
        dep_feature = self.dep_embed(dep_tags)
        if self.args.highway:
            dep_feature = self.highway_dep(dep_feature)

        # fmask是前面为1，后面的aspect为0
        dep_out = [g(feature, dep_feature, fmask).unsqueeze(1) for g in self.gat_dep]  # (N, 1, D) * num_heads
        dep_out = torch.cat(dep_out, dim=1)  # (N, H, D)
        dep_out = dep_out.mean(dim=1)  # (N, D)


        feature_out = torch.cat([dep_out,  pool_out], dim=1)  # (N, D')
        # feature_out = gat_out
        #############################################################################################
        x = self.dropout(feature_out)
        x = self.fcs(x)
        logit = self.fc_final(x)
        return logit


def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True, use_cuda=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    if use_cuda:
        return h0.cuda(), c0.cuda()
    else:
        return h0, c0
