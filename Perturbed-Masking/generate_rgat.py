import argparse
import warnings

warnings.filterwarnings("ignore")


import json
import os

from dependency import _evaluation as dep_eval
from dependency.dep_parsing import decoding_new as dep_parsing_new

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--matrix_folder", default="bert/Laptop")
    parser.add_argument("--layers", default="11")

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
            "left_branching",
            "gold",
        ],
    )
    parser.add_argument("--weight", default=0.0, type=float)

    args = parser.parse_args()
    print(args)

    # trees: [[(w_i, head_i), (w_i, head_i), ...], []]
    # results: [[line, 0, 0], [line, 0, 0]]

    model_type, dataset = args.matrix_folder.split("/")
    matrix_folder = "/your/work/space/save_matrix/" + args.matrix_folder

    os.makedirs(os.path.join("/your/work/space/rgat", model_type, args.layers, dataset), exist_ok=True)
    save_folder = os.path.join("/your/work/space/rgat", model_type, args.layers, dataset)
    print("Save to {}".format(save_folder))

    fns = os.listdir(matrix_folder)
    for fn in fns:
        ps = fn[:-4].split("-")
        layer = ps[-1]
        if layer == args.layers:
            split = ps[-2].lower()
            args.matrix = os.path.join(matrix_folder, fn)
            trees, results = dep_parsing_new(args)
            # print(trees[0])
            # exit()
            dep_eval(trees, results)
            assert len(trees) == len(results), (len(trees), len(results))
            # trees: [[(w_i, head_i), (w_i, head_i), ...], []]
            # results: [[line, 0, 0], [line, 0, 0]]
            dicts = []
            for ((line, _, _), tree) in zip(results, trees):
                if args.layers != "0":
                    tree = [int(head) for (w_i, head) in tree[1:]]  # 去掉cls
                else:
                    tree = [x.head for x in line][1:]  # 删掉root
                d = {}
                d["tokens"] = [x.form for x in line][1:]
                d["sentence"] = " ".join(d["tokens"])
                d["head"] = tree
                d["tags"] = ["NNP"] * len(d["head"])
                d["predicted_dependencies"] = ["nn"] * len(d["head"])
                d["predicted_heads"] = tree
                d["dependencies"] = [
                    ["nn", h, idx] for idx, h in enumerate(tree, start=1)
                ]
                aspects = line[0].aspects
                d["aspect_sentiment"] = []
                d["from_to"] = []
                for aspect in aspects:
                    d["aspect_sentiment"].append(
                        [" ".join(aspect["term"]), aspect["polarity"]]
                    )
                    d["from_to"].append([aspect["from"], aspect["to"]])
                assert len(d["tokens"]) == len(tree)
                dicts.append(d)
            with open(
                os.path.join(
                    save_folder, "{}_{}.json".format(dataset, split.capitalize())
                ),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(dicts, f)
