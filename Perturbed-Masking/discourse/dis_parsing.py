import argparse
from tqdm import tqdm
import numpy as np
import pickle
from dependency import Eisner
from dependency import chuliu_edmonds
from .dis_eval import evaluation, distance_evaluation


def find_root(parse):
    for edu in parse:
        if edu['parent'] == 0:
            return edu['id']
    return False


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1)


def decoding(args):
    trees = []
    deprels = []
    with open(args.matrix, 'rb') as f:
        results = pickle.load(f)
    gold_trees = []
    decoder = Eisner()
    root_found = 0

    for (gold_tree, init_matrix) in tqdm(results):
        root = find_root(gold_tree)
        gold_trees.append(gold_tree)

        # cut SEP and transpose to the original matrix
        final_matrix = np.array(init_matrix[:-1,:-1]).transpose()
        final_matrix = final_matrix.transpose()

        if final_matrix.shape[0] == 0:
            print('find empty matrix:')
            continue
        assert final_matrix.shape[0] == final_matrix.shape[1]

        if args.decoder == 'cle':
            final_matrix[0] = 0
            if root and args.root == 'gold':
                final_matrix[root] = 0
                final_matrix[root, 0] = 1
            # final_matrix /= np.sum(final_matrix, axis=1, keepdims=True)

            best_heads = chuliu_edmonds(final_matrix)
            trees.append([(i, head) for i, head in enumerate(best_heads)])

        if args.decoder == 'eisner':
            if root and args.root == 'gold':
                final_matrix[root] = 0
                final_matrix[root, 0] = 1
            # final_matrix = softmax(final_matrix)
            final_matrix = final_matrix.transpose()

            best_heads, _ = decoder.parse_proj(final_matrix)
            trees.append([(i, head) for i, head in enumerate(best_heads)])
        if args.decoder == 'right_chain':
            trees.append([(root, 0) if i == root else (i, i + 1) for i in range(0, final_matrix.shape[0])])
        if args.decoder == 'left_chain':
            trees.append([(root, 0) if i == root else (i, i - 1) for i in range(0, final_matrix.shape[0])])
            # trees.append([ (i, i - 1) for i in range(0, final_matrix.shape[0])])

    return trees, gold_trees, deprels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data args
    parser.add_argument('--matrix', default='../results/discourse/bert-dist-SciDTB-last.pkl')

    # Decoding args
    parser.add_argument('--decoder', default='cle')
    parser.add_argument('--root', default='gold', help='gold or cls')

    args = parser.parse_args()

    trees, gold_trees, deprels = decoding(args)
    evaluation(trees, gold_trees)
    distance_evaluation(trees, gold_trees)