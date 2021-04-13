import argparse
import warnings

warnings.filterwarnings("ignore")
import pickle

from dependency.dep_parsing import decoding_new as dep_parsing_new
import networkx as nx
from dependency import _evaluation as dep_eval
import os
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--matrix_folder", default="bert/Restaurants", help="bert/Restaurants"
    )
    parser.add_argument("--layers", default="12")

    parser.add_argument("--subword", default="avg", choices=["first", "avg", "max"])
    parser.add_argument("--root", default="non-gold", help="use gold root as init")
    parser.add_argument(
        "--decoder",
        default="cle",
        choices=[
            "eisner",
            "cle",
            "right_chain",
            "top_down",
            "mart",
            "right_branching",
            "left_chain",
            "gold",
        ],
    )

    args = parser.parse_args()
    print(args)

    model_type, dataset = args.matrix_folder.split("/")
    matrix_folder = "save_matrix/" + args.matrix_folder
    os.makedirs(os.path.join("asgcn2", model_type), exist_ok=True)
    os.makedirs(os.path.join("asgcn2", model_type, args.layers), exist_ok=True)
    os.makedirs(os.path.join("asgcn2", model_type, args.layers, dataset), exist_ok=True)
    save_folder = os.path.join("asgcn2", model_type, args.layers, dataset)

    print("Save to {}".format(save_folder))
    mapping = {"positive": 1, "neutral": 0, "negative": -1}
    fns = os.listdir(matrix_folder)
    for fn in fns:
        ps = fn[:-4].split("-")
        layer = ps[-1]
        if layer == args.layers:
            split = ps[-2]  # train or test
            split = split[0].capitalize() + split[1:].lower()
            args.matrix = os.path.join(matrix_folder, fn)
            if split == "Test":
                gold_fn = "{}_{}_Gold.xml.seg".format(dataset, split)
                graph_fn = "{}_{}_Gold.xml.seg.graph".format(dataset, split)
            else:
                gold_fn = "{}_{}.xml.seg".format(dataset, split)
                graph_fn = "{}_{}.xml.seg.graph".format(dataset, split)
            trees, results = dep_parsing_new(args)
            dep_eval(trees, results)
            with open(
                os.path.join(save_folder, gold_fn), "w", encoding="utf8"
            ) as f1, open(os.path.join(save_folder, graph_fn), "wb") as f2:
                print("Writing to {}".format(gold_fn))
                adj_matrixes = {}
                for ((line, _, _), tree) in zip(results, trees):
                    sentence = [x.form for x in line][1:]  # [去掉root]
                    # w_i-1是因为有cls
                    if args.layers != "0":
                        edges = [
                            (head - 1, w_i - 1) if head != 0 else (w_i - 1, w_i - 1)
                            for (w_i, head) in tree[1:]
                        ]
                    else:
                        edges = [
                            (x.id - 1, x.head - 1)
                            if x.head != 0
                            else (x.id - 1, x.id - 1)
                            for x in line
                        ][1:]
                    assert len(sentence) == len(tree) - 1
                    aspects = line[0].aspects
                    # f1需要存的是
                    adj_matrix = np.eye(len(sentence)).astype("float32")
                    for i in range(len(sentence)):
                        for i, j in edges:
                            adj_matrix[i, j] = 1
                            adj_matrix[j, i] = 1
                    for aspect in aspects:
                        terms = aspect["term"]
                        tokens = (
                            sentence[: aspect["from"]]
                            + ["$T$"]
                            + sentence[aspect["to"] :]
                        )
                        f1.write(" ".join(tokens) + "\n")
                        f1.write(" ".join(terms) + "\n")
                        f1.write("{}\n".format(mapping[aspect["polarity"]]))
                        adj_matrixes[len(adj_matrixes) * 3] = adj_matrix
                pickle.dump(adj_matrixes, f2)
