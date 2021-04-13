# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np


def load_word_vec(path, word2idx=None, embed_dim=300):
    fin = open(path, "r", encoding="utf-8", newline="\n", errors="ignore")
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        word, vec = " ".join(tokens[:-embed_dim]), tokens[-embed_dim:]
        if word in word2idx.keys():
            word_vec[word] = np.asarray(vec, dtype="float32")
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, type, refresh=0):
    # 这里的type就是某种数据集的名称
    embedding_matrix_file_name = "{0}_{1}_embedding_matrix.pkl".format(
        str(embed_dim), type
    )
    if os.path.exists(embedding_matrix_file_name) and refresh == 0:
        print("loading embedding_matrix:", embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, "rb"))
    else:
        print("loading new word vectors...")
        embedding_matrix = np.zeros((len(word2idx), embed_dim))
        embedding_matrix[1, :] = np.random.uniform(
            -1 / np.sqrt(embed_dim), 1 / np.sqrt(embed_dim), (1, embed_dim)
        )
        if type == "fr":
            fname = "../glove/word2vec_french.txt"
        elif type == "dutch":
            fname = "../glove/word2vec_dutch.txt"
        elif type == "sp":
            fname = "../glove/word2vec_spanish.txt"
        else:
            fname = "../../glove/glove.840B.300d.txt"
        word_vec = load_word_vec(fname, word2idx=word2idx)
        print("building embedding_matrix:", embedding_matrix_file_name)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, "wb"))
    return embedding_matrix


class Tokenizer(object):
    def __init__(self, word2idx=None):
        if word2idx is None:
            self.word2idx = {}
            self.idx2word = {}
            self.idx = 0
            self.word2idx["<pad>"] = self.idx
            self.idx2word[self.idx] = "<pad>"
            self.idx += 1
            self.word2idx["<unk>"] = self.idx
            self.idx2word[self.idx] = "<unk>"
            self.idx += 1
        else:
            self.word2idx = word2idx
            self.idx2word = {v: k for k, v in word2idx.items()}

    def fit_on_text(self, text):
        text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text):
        text = text.lower()
        words = text.split()
        unknownidx = 1
        sequence = [
            self.word2idx[w] if w in self.word2idx else unknownidx for w in words
        ]
        if len(sequence) == 0:
            sequence = [0]
        return sequence


class ABSADataset(object):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class ABSADatesetReader:
    @staticmethod
    def __read_text__(fnames):
        text = ""
        for fname in fnames:
            fin = open(fname, "r", encoding="utf-8", newline="\n", errors="ignore")
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [
                    s.lower().strip() for s in lines[i].partition("$T$")
                ]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "
        return text

    @staticmethod
    def __read_data__(fname, tokenizer):
        fin = open(fname, "r", encoding="utf-8", newline="\n", errors="ignore")
        lines = fin.readlines()
        fin.close()
        with open(fname + ".graph", "rb") as fin:
            idx2gragh = pickle.load(fin)  
        all_data = []
        for i in range(0, len(lines), 3):
            text_left, _, text_right = [
                s.lower().strip() for s in lines[i].partition("$T$")
            ]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()

            text_indices = tokenizer.text_to_sequence(
                text_left + " " + aspect + " " + text_right
            )
            context_indices = tokenizer.text_to_sequence(text_left + " " + text_right)
            aspect_indices = tokenizer.text_to_sequence(aspect)
            left_indices = tokenizer.text_to_sequence(text_left)
            polarity = int(polarity) + 1
            dependency_graph = idx2gragh[i]

            data = {
                "text_indices": text_indices,
                "context_indices": context_indices,
                "aspect_indices": aspect_indices,
                "left_indices": left_indices,
                "polarity": polarity,
                "dependency_graph": dependency_graph,
            }

            all_data.append(data)
        return all_data

    def __init__(self, dataset="twitter", embed_dim=300, refresh=False):
        print("preparing {0} dataset ...".format(dataset))
        if "/" not in dataset:
            fname = {
                "Tweets": {
                    "train": "./datasets/acl-14-short-data/train.raw",
                    "test": "./datasets/acl-14-short-data/test.raw",
                },
                "Restaurants": {
                    "train": "./datasets/semeval14/restaurant_train.raw",
                    "test": "./datasets/semeval14/restaurant_test.raw",
                },
                "Laptop": {
                    "train": "./datasets/semeval14/laptop_train.raw",
                    "test": "./datasets/semeval14/laptop_test.raw",
                },
                "Rest15": {
                    "train": "./datasets/semeval15/restaurant_train.raw",
                    "test": "./datasets/semeval15/restaurant_test.raw",
                },
                "Rest16": {
                    "train": "./datasets/semeval16/restaurant_train.raw",
                    "test": "./datasets/semeval16/restaurant_test.raw",
                },
            }
            fname = fname[dataset]
        else:
            fns = os.listdir(dataset)
            train_fp = os.path.join(
                dataset, [fn for fn in fns if "Train" in fn and fn.endswith("seg")][0]
            )
            test_fp = os.path.join(
                dataset, [fn for fn in fns if "Test" in fn and fn.endswith("seg")][0]
            )
            fname = {"train": train_fp, "test": test_fp}

        text = ABSADatesetReader.__read_text__([fname["train"], fname["test"]])
        dataset = os.path.basename(dataset)
        if os.path.exists(dataset + "_word2idx.pkl") and not refresh:
            print("loading {0} tokenizer...".format(dataset))
            with open(dataset + "_word2idx.pkl", "rb") as f:
                word2idx = pickle.load(f)
                tokenizer = Tokenizer(word2idx=word2idx)
        else:
            tokenizer = Tokenizer()
            tokenizer.fit_on_text(text)
            with open(dataset + "_word2idx.pkl", "wb") as f:
                pickle.dump(tokenizer.word2idx, f)
        self.embedding_matrix = build_embedding_matrix(
            tokenizer.word2idx, embed_dim, dataset, refresh
        )
        self.train_data = ABSADataset(
            ABSADatesetReader.__read_data__(fname["train"], tokenizer)
        )
        self.test_data = ABSADataset(
            ABSADatesetReader.__read_data__(fname["test"], tokenizer)
        )
