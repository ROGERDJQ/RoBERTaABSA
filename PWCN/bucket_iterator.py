# -*- coding: utf-8 -*-

import math
import random
import torch

class BucketIterator(object):
    def __init__(self, data, batch_size, shuffle=True, sort=True):
        self.shuffle = shuffle
        self.sort = sort
        self.batches = self.sort_and_pad(data, batch_size)
        self.batch_len = len(self.batches)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        if self.sort:
            sorted_data = sorted(data, key=lambda x: len(x['text_indices']))
        else:
            sorted_data = data
        batches = []
        for i in range(num_batch):
            batches.append(self.pad_data(sorted_data[i*batch_size : (i+1)*batch_size]))
        return batches

    @staticmethod
    def pad_data(batch_data):
        batch_text_indices = []
        batch_aspect_indices = []
        batch_left_indices = []
        batch_polarity = []
        batch_dependency_dist = []
        max_len = max([len(t['text_indices']) for t in batch_data])
        for item in batch_data:
            text_indices, aspect_indices, left_indices, polarity, dependency_dist = \
                item['text_indices'], item['aspect_indices'], item['left_indices'],\
                item['polarity'], item['dependency_dist']
            text_padding = [0] * (max_len - len(text_indices))
            aspect_padding = [0] * (max_len - len(aspect_indices))
            left_padding = [0] * (max_len - len(left_indices))
            dependency_dist_padding = [0] * (max_len - len(dependency_dist))
            batch_text_indices.append(text_indices + text_padding)
            batch_aspect_indices.append(aspect_indices + aspect_padding)
            batch_left_indices.append(left_indices + left_padding)
            batch_polarity.append(polarity)
            batch_dependency_dist.append(dependency_dist + dependency_dist_padding)
        return { \
                'text_indices': torch.tensor(batch_text_indices), \
                'aspect_indices': torch.tensor(batch_aspect_indices), \
                'left_indices': torch.tensor(batch_left_indices), \
                'polarity': torch.tensor(batch_polarity), \
                'dependency_dist': torch.tensor(batch_dependency_dist)
                }

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for idx in range(self.batch_len):
            yield self.batches[idx]
