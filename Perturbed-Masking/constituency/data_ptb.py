import os
import re
from tqdm import tqdm
import nltk
from nltk.corpus import ptb
from nltk.tree import ParentedTree

word_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
             'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
             'WDT', 'WP', 'WP$', 'WRB']
currency_tags_words = ['#', '$', 'C$', 'A$']
ellipsis = ['*', '*?*', '0', '*T*', '*ICH*', '*U*', '*RNR*', '*EXP*', '*PPA*', '*NOT*']
punctuation_tags = ['.', ',', ':', '-LRB-', '-RRB-', '\'\'', '``']
punctuation_words = ['.', ',', ':', '-LRB-', '-RRB-', '\'\'', '``', '--', ';', '-', '?', '!', '...', '-LCB-', '-RCB-']


class Corpus(object):
    def __init__(self, path, split='WSJ10'):

        file_ids = []
        for one_day in os.listdir(path):
            if one_day.endswith('.DS_Store'):
                continue
            for one_article in os.listdir(os.path.join(path, one_day)):
                file_ids.append('WSJ/' + str(one_day) + '/' + one_article)
        train_file_ids = []
        test_file_ids = []
        for id in file_ids:
            if 'WSJ/00/WSJ_0000.MRG' <= id <= 'WSJ/24/WSJ_2499.MRG':
                train_file_ids.append(id)
            if 'WSJ/23/WSJ_2300.MRG' <= id <= 'WSJ/23/WSJ_2399.MRG':
                test_file_ids.append(id)
        if split == 'WSJ10':
            self.sens, self.trees, self.nltk_trees = self.tokenize(train_file_ids, True)
        elif split == 'WSJ23':
            self.sens, self.trees, self.nltk_trees = self.tokenize(test_file_ids, False)
        print(len(self.nltk_trees))

    def filter_words(self, tree):
        words = []
        for w, tag in tree.pos():
            if tag in word_tags:
                w = w.lower()
                w = re.sub('[0-9]+', 'N', w)
                if w == 'N.N' or w == 'N\\/N' or w == 'N,N' or w == 'N,N,N' or w == 'N-N' :
                    w = '19'
                # if tag == 'CD':
                #     w = 'N'
                words.append(w)
        return words

    def tokenize(self, file_ids, wsj10):

        def tree2list(tree):
            if isinstance(tree, nltk.Tree):
                if tree.label() in word_tags:
                    return tree.leaves()[0]
                else:
                    root = []
                    for child in tree:
                        c = tree2list(child)
                        if c != []:
                            root.append(c)
                    if len(root) > 1:
                        return root
                    elif len(root) == 1:
                        return root[0]
            return []

        def tree2tree_wo_punc(old_tree):
            tree = ParentedTree.convert(old_tree)
            for sub in reversed(list(tree.subtrees())):
                if sub.height() == 2 and sub.label() not in word_tags:  # find not word subtree
                    parent = sub.parent()
                    while parent and len(parent) == 1:
                        sub = parent
                        parent = sub.parent()
                    print(sub, "will be deleted")
                    del tree[sub.treeposition()]
            return tree

        sens = []
        trees = []
        nltk_trees = []
        for id in tqdm(file_ids):
            sentences = ptb.parsed_sents(id)
            for sen_tree in sentences:
                words = self.filter_words(sen_tree)
                if len(words) > 10 and wsj10:
                    continue
                sens.append(words)
                trees.append(tree2list(sen_tree))
                nltk_trees.append(sen_tree)

        return sens, trees, nltk_trees


# corpus = Corpus('./data/WSJ', 'WSJ23')