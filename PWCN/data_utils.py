# -*- coding: utf-8 -*-

import os
import pickle
import random
import numpy as np
import networkx as nx
import json
from networkx import NetworkXNoPath


def load_word_vec(path, word2idx=None):
    fin = open(path, "r", encoding="utf-8", newline="\n", errors="ignore")
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx:
            word_vec[tokens[0]] = np.asarray(tokens[-300:], dtype="float32")
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, type, refresh=False):
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
            fname = "../../glove/word2vec_spanish.txt"
        else:
            fname = "../glove/glove.840B.300d.txt"
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
        fin = open(
            fname + ".dist", "r", encoding="utf-8", newline="\n", errors="ignore"
        )
        dist_lines = fin.readlines()
        fin.close()

        all_data = []
        cnt = 0
        for i in range(0, len(lines), 3):
            text_left, _, text_right = [
                s.lower().strip() for s in lines[i].partition("$T$")
            ]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()

            text_indices = tokenizer.text_to_sequence(
                text_left + " " + aspect + " " + text_right
            )
            aspect_indices = tokenizer.text_to_sequence(aspect)
            left_indices = tokenizer.text_to_sequence(text_left)
            polarity = int(polarity) + 1
            dependency_dist = [float(d) for d in dist_lines[cnt * 2 + 1].split()]
            cnt += 1

            data = {
                "text_indices": text_indices,
                "aspect_indices": aspect_indices,
                "left_indices": left_indices,
                "polarity": polarity,
                "dependency_dist": dependency_dist,
            }

            all_data.append(data)
        return all_data

    def __init__(self, dataset="laptop", embed_dim=300, refresh=False):
        print("preparing {0} dataset...".format(dataset))
        if "/" not in dataset:
            fname = {
                "restaurant": {
                    "train": "./datasets/semeval14/Restaurants_Train.xml.seg",
                    "test": "./datasets/semeval14/Restaurants_Test_Gold.xml.seg",
                },
                "laptop": {
                    "train": "./datasets/semeval14/Laptops_Train.xml.seg",
                    "test": "./datasets/semeval14/Laptops_Test_Gold.xml.seg",
                },
            }
            train_fp = fname[dataset]["train"]
            test_fp = fname[dataset]["test"]
            text = ABSADatesetReader.__read_text__(
                [fname[dataset]["train"], fname[dataset]["test"]]
            )
        else:
            # 需要生成一下
            fns = os.listdir(dataset)
            length = len(list(filter(lambda x: x.endswith("seg"), fns)))
            assert (
                length < 3
            ), f"You should only have two files ends with .seg in {dataset}"
            regenerate = True
            if length == 2:
                # 说明曾经已经生成过数据了，这里check一下防止出现没有update的情况
                regenerate = False
                train_fp = os.path.join(
                    dataset,
                    [fn for fn in fns if "Train" in fn and fn.endswith("seg")][0],
                )
                seg_train_fp_m_time = os.path.getmtime(train_fp)
                if os.path.exists(train_fp[:-4]):
                    train_fp_m_time = os.path.getmtime(train_fp[:-4])
                    if seg_train_fp_m_time < train_fp_m_time:
                        regenerate = True
                test_fp = os.path.join(
                    dataset,
                    [fn for fn in fns if "Test" in fn and fn.endswith("seg")][0],
                )
                seg_test_fp_m_time = os.path.getmtime(test_fp)
                if os.path.exists(test_fp[:-4]):
                    test_fp_m_time = os.path.getmtime(test_fp[:-4])
                    if seg_test_fp_m_time < test_fp_m_time:
                        regenerate = True

            if regenerate:
                print(f"Generate PWCN data for {dataset}")
                train_fp, test_fp = generate_seg_dist_from_raw(dataset)

            text = ABSADatesetReader.__read_text__([train_fp, test_fp])

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
            ABSADatesetReader.__read_data__(train_fp, tokenizer)
        )
        self.test_data = ABSADataset(
            ABSADatesetReader.__read_data__(test_fp, tokenizer)
        )


def generate_seg_dist_from_raw(folder):
    fns = os.listdir(folder)
    raw_train_fp = os.path.join(folder, [fn for fn in fns if "Train" in fn][0])
    raw_test_fp = os.path.join(folder, [fn for fn in fns if "Test" in fn][0])
    convert_data(raw_train_fp)
    convert_data(raw_test_fp)
    return raw_train_fp + ".seg", raw_test_fp + ".seg"


mapping = {"positive": 1, "neutral": 0, "negative": -1}


def convert_data(fp):
    seg_fp = fp + ".seg"
    dist_fp = seg_fp + ".dist"
    with open(fp, "r", encoding="utf-8") as f:
        samples = json.load(f)
    with open(seg_fp, "w", encoding="utf8") as f1, open(
        dist_fp, "w", encoding="utf8"
    ) as f2:
        for sample in samples:
            heads = sample["head"]
            edges = [
                (head - 1, w_i - 1) if head != 0 else (w_i - 1, w_i - 1)
                for (w_i, head) in enumerate(heads, start=1)
            ]
            graph = nx.Graph(edges)
            sentence = sample["token"]
            aspects = sample["aspects"]
            for aspect in aspects:
                terms = aspect["term"]
                tokens = sentence[: aspect["from"]] + ["$T$"] + sentence[aspect["to"] :]
                f1.write(" ".join(tokens) + "\n")
                f1.write(" ".join(terms) + "\n")
                f1.write("{}\n".format(mapping[aspect["polarity"]]))
                f2.write(" " + " ".join(tokens) + "\n")
                distances = []
                for i in range(len(sentence)):
                    if aspect["from"] <= i < aspect["to"]:
                        distances.append(0.0)
                    else:
                        min_dis = len(sentence) / 2.0
                        for asp in range(aspect["from"], aspect["to"]):
                            try:
                                dis = nx.shortest_path_length(
                                    graph, source=asp, target=i
                                )
                            except NetworkXNoPath:
                                dis = len(sentence) / 2.0
                            if dis < min_dis:
                                min_dis = dis
                        distances.append(min_dis)
                f2.write(" ".join(map(str, distances)) + "\n")
