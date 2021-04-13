import numpy as np


def find_best_cut(scores):
    best_score = np.inf
    best_cut = -1
    for k in range(1, len(scores)):
        sq1 = 2 * k
        sq2 = 2 * (len(scores) - k)
        rec = (len(scores) - k) * k
        left = np.sum(scores[:k, :k]) / sq1
        right = np.sum(scores[k + 1:, k + 1:]) / sq2
        between = np.sum(scores[:k, k + 1:]) + np.sum(scores[k + 1:, :k])
        between /= rec
        cut_score = left + right - between
        if cut_score < best_score:
            best_cut = k
            best_score = cut_score
    return best_cut


def mart(scores, sen):
    assert len(scores) == len(sen)

    if len(scores) == 1:
        parse_tree = sen[0]
    else:
        idx_max = find_best_cut(scores)
        parse_tree = []
        if len(sen[:idx_max]) > 0:
            tree0 = mart(scores[:idx_max, :idx_max], sen[:idx_max])
            parse_tree.append(tree0)
        tree1 = sen[idx_max]
        if len(sen[idx_max + 1:]) > 0:
            tree2 = mart(scores[idx_max + 1:, idx_max + 1:], sen[idx_max + 1:])
            tree1 = [tree1, tree2]
        if parse_tree == []:
            parse_tree = tree1
        else:
            parse_tree.append(tree1)
    return parse_tree


def right_branching(sent):
    if type(sent) is not list:
        return sent
    if len(sent) == 1:
        return sent[0]
    else:
        return [sent[0], right_branching(sent[1:])]


def left_branching(sent):
    if type(sent) is not list:
        return sent
    if len(sent) == 1:
        return sent[0]
    else:
        return [ left_branching(sent[:-1]), sent[-1]]

