import os
import json
import codecs


class Corpus(object):
    def __init__(self, file_dir, tokenizer):
        self.tokenizer = tokenizer
        self.scidtb = []
        for file in os.listdir(file_dir):
            json_file = codecs.open(file_dir+file, 'r', 'utf-8-sig')
            tree = json.loads(json_file.read())['root']
            for edu in tree:
                if edu['text'].endswith('<S>'):
                    edu['text'] = edu['text'][:-3]
                edu['tokens'] = self.tokenizer.tokenize(edu['text'])
            tree[0]['tokens'] = ['[CLS]']
            for edu in tree:
                edu['tokens_id'] = self.tokenizer.convert_tokens_to_ids(edu['tokens'])
            batch, batch_pos  = self.tree2batch(tree)
            self.scidtb.append((batch, batch_pos, tree))
        print('get corpus {}, done'.format(file_dir))

    def tree2batch(self, tree):
        batch = []
        batch_pos = []
        for i in range(len(tree)):
            one_batch_doc = []
            for j in range(len(tree)):
                doc = self.get_whole_doc(tree, i, j)
                one_batch_doc.append(doc)
            one_batch_pos = []
            for k in range(len(tree)):
                if k != i:
                    one_batch_pos += [0] * len(tree[k]['tokens_id'])
                else:
                    one_batch_pos += [1] * len(tree[k]['tokens_id'])
            one_batch_pos.append(0)
            batch.append(one_batch_doc)
            batch_pos.append(one_batch_pos)
        return batch, batch_pos

    def get_whole_doc(self, tree, index_i, index_j):
        mask_id = self.tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
        sep_id = self.tokenizer.convert_tokens_to_ids(['[SEP]'])[0]
        doc = []
        for k in range(len(tree)):
            if k != index_i and k != index_j:
                doc += tree[k]['tokens_id']
            else:
                doc += [mask_id]*len(tree[k]['tokens_id'])

        doc.append(sep_id)
        return doc

    def token2ids(self, batch):
        tokenized_batch = []
        for sample in batch:
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(sample)
            tokenized_batch.append(indexed_tokens)
        return tokenized_batch

# from transformers import BertTokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# corpus = Corpus('SciDTB/test/gold/', tokenizer)