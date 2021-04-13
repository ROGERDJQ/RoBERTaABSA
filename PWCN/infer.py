# -*- coding: utf-8 -*-


import os
import pickle
import torch
import torch.nn.functional as F
import argparse

from data_utils import ABSADatesetReader, ABSADataset, Tokenizer, build_embedding_matrix
from bucket_iterator import BucketIterator
from models import LSTM, PWCN_POS, PWCN_DEP
from dependency_dist import dependency_dist_func

class Inferer:
    """A simple inference example"""
    def __init__(self, opt):
        self.opt = opt
        print("loading {0} tokenizer...".format(opt.dataset))
        with open(opt.dataset+'_word2idx.pkl', 'rb') as f:
            word2idx = pickle.load(f)
            self.tokenizer = Tokenizer(word2idx=word2idx)
        embedding_matrix = build_embedding_matrix(self.tokenizer.word2idx, opt.embed_dim, opt.dataset)
        self.model = opt.model_class(embedding_matrix, opt)
        print('loading model {0} ...'.format(opt.model_name))
        self.model.load_state_dict(torch.load(opt.state_dict_path, map_location=lambda storage, loc: storage))
        # switch model to evaluation mode
        self.model.eval()
        torch.autograd.set_grad_enabled(False)

    def evaluate(self, raw_text, aspect):
        text_seqs = [self.tokenizer.text_to_sequence(raw_text.lower().strip())]
        aspect_seqs = [self.tokenizer.text_to_sequence(aspect.lower())]
        left_seqs = [self.tokenizer.text_to_sequence(raw_text.lower().split(aspect.lower())[0])]
        text_indices = torch.tensor(text_seqs, dtype=torch.int64)
        aspect_indices = torch.tensor(aspect_seqs, dtype=torch.int64)
        left_indices = torch.tensor(left_seqs, dtype=torch.int64)
        dependency_dist = torch.tensor([dependency_dist_func(raw_text, aspect)], dtype=torch.int64)
        data = {
            'text_indices':text_indices, 
            'aspect_indices':aspect_indices, 
            'left_indices':left_indices,
            'dependency_dist':dependency_dist,
        }
        t_inputs = [data[col] for col in self.opt.inputs_cols]
        t_outputs = self.model(t_inputs)

        t_probs = F.softmax(t_outputs, dim=-1).cpu().numpy()
        return t_probs

if __name__ == '__main__':
    model_classes = {
        'lstm': LSTM,
        'pwcn_pos': PWCN_POS,
        'pwcn_dep': PWCN_DEP,
    }
    dataset = 'restaurant'
    # set your trained models here
    model_state_dict_paths = {
        'lstm': 'state_dict/lstm_'+dataset+'.pkl',
        'pwcn_pos': 'state_dict/pwcn_pos_'+dataset+'.pkl',
        'pwcn_dep': 'state_dict/pwcn_dep_'+dataset+'.pkl',
    }
    input_colses = {
        'lstm': ['text_indices'],
        'pwcn_pos': ['text_indices', 'aspect_indices', 'left_indices'], 
        'pwcn_dep': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_dist'],
    }
    class Option(object): pass
    opt = Option()
    opt.model_name = 'pwcn_dep'
    opt.model_class = model_classes[opt.model_name]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.dataset = dataset
    opt.state_dict_path = model_state_dict_paths[opt.model_name]
    opt.embed_dim = 300
    opt.hidden_dim = 300
    opt.polarities_dim = 3
    opt.device = torch.device('cpu')

    inf = Inferer(opt)
    t_probs = inf.evaluate('great food but the service was dreadful !', 'food')
    print(t_probs.argmax(axis=-1))
