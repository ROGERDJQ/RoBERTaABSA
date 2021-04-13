# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
import spacy
nlp = spacy.load('en_core_web_sm')

def dependency_dist_func(text, aspect_term):
    # https://spacy.io/docs/usage/processing-text
    document = nlp(text)
    # Load spacy's dependency tree into a networkx graph
    edges = []
    
    for token in document:
        # FYI https://spacy.io/docs/api/token
        for child in token.children:
            edges.append((token.i, child.i))
    graph = nx.Graph(edges)

    text_lst = text.split()
    seq_len = len(text_lst)
    text_left, _, _ = text.partition(aspect_term)
    start = len(text_left.split())
    end = start + len(aspect_term.split())
    asp_idx = [i for i in range(start, end)]
    dist_matrix = seq_len*np.ones((seq_len, len(asp_idx))).astype('float32')
    for i, asp in enumerate(asp_idx):
        for j in range(seq_len):
            try:
                dist_matrix[j][i] = nx.shortest_path_length(graph, source=asp, target=j)
            except:
                dist_matrix[j][i] = seq_len/2
    dist_matrix = np.min(dist_matrix, axis=1)
    return dist_matrix

