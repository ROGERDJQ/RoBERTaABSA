import numpy as np
from collections import defaultdict


# ***************************************************************
def tarjan(tree):
    """"""

    indices = -np.ones_like(tree)
    lowlinks = -np.ones_like(tree)
    onstack = np.zeros_like(tree, dtype=bool)
    stack = list()
    _index = [0]
    cycles = []

    # -------------------------------------------------------------
    def strong_connect(i):
        _index[0] += 1
        index = _index[-1]
        indices[i] = lowlinks[i] = index - 1
        stack.append(i)
        onstack[i] = True
        dependents = np.where(np.equal(tree, i))[0]
        for j in dependents:
            if indices[j] == -1:
                strong_connect(j)
                lowlinks[i] = min(lowlinks[i], lowlinks[j])
            elif onstack[j]:
                lowlinks[i] = min(lowlinks[i], indices[j])

        # There's a cycle!
        if lowlinks[i] == indices[i]:
            cycle = np.zeros_like(indices, dtype=bool)
            while stack[-1] != i:
                j = stack.pop()
                onstack[j] = False
                cycle[j] = True
            stack.pop()
            onstack[i] = False
            cycle[i] = True
            if cycle.sum() > 1:
                cycles.append(cycle)
        return

    # -------------------------------------------------------------
    for i in range(len(tree)):
        if indices[i] == -1:
            strong_connect(i)
    return cycles


# ===============================================================
def chuliu_edmonds(scores):
    """"""

    scores *= (1 - np.eye(scores.shape[0]))
    scores[0] = 0
    scores[0, 0] = 1
    tree = np.argmax(scores, axis=1)
    cycles = tarjan(tree)
    # print(scores)
    # print(cycles)
    if not cycles:
        return tree
    else:
        # t = len(tree); c = len(cycle); n = len(noncycle)
        # locations of cycle; (t) in [0,1]
        cycle = cycles.pop()
        # indices of cycle in original tree; (c) in t
        cycle_locs = np.where(cycle)[0]
        # heads of cycle in original tree; (c) in t
        cycle_subtree = tree[cycle]
        # scores of cycle in original tree; (c) in R
        cycle_scores = scores[cycle, cycle_subtree]
        # total score of cycle; () in R
        cycle_score = cycle_scores.prod()

        # locations of noncycle; (t) in [0,1]
        noncycle = np.logical_not(cycle)
        # indices of noncycle in original tree; (n) in t
        noncycle_locs = np.where(noncycle)[0]
        # print(cycle_locs, noncycle_locs)

        # scores of cycle's potential heads; (c x n) - (c) + () -> (n x c) in R
        metanode_head_scores = scores[cycle][:, noncycle] / cycle_scores[:, None] * cycle_score
        # scores of cycle's potential dependents; (n x c) in R
        metanode_dep_scores = scores[noncycle][:, cycle]
        # best noncycle head for each cycle dependent; (n) in c
        metanode_heads = np.argmax(metanode_head_scores, axis=0)
        # best cycle head for each noncycle dependent; (n) in c
        metanode_deps = np.argmax(metanode_dep_scores, axis=1)

        # scores of noncycle graph; (n x n) in R
        subscores = scores[noncycle][:, noncycle]
        # pad to contracted graph; (n+1 x n+1) in R
        subscores = np.pad(subscores, ((0, 1), (0, 1)), 'constant')
        # set the contracted graph scores of cycle's potential heads; (c x n)[:, (n) in n] in R -> (n) in R
        subscores[-1, :-1] = metanode_head_scores[metanode_heads, np.arange(len(noncycle_locs))]
        # set the contracted graph scores of cycle's potential dependents; (n x c)[(n) in n] in R-> (n) in R
        subscores[:-1, -1] = metanode_dep_scores[np.arange(len(noncycle_locs)), metanode_deps]

        # MST with contraction; (n+1) in n+1
        contracted_tree = chuliu_edmonds(subscores)
        # head of the cycle; () in n
        # print(contracted_tree)
        cycle_head = contracted_tree[-1]
        # fixed tree: (n) in n+1
        contracted_tree = contracted_tree[:-1]
        # initialize new tree; (t) in 0
        new_tree = -np.ones_like(tree)
        # print(0, new_tree)
        # fixed tree with no heads coming from the cycle: (n) in [0,1]
        contracted_subtree = contracted_tree < len(contracted_tree)
        # add the nodes to the new tree (t)[(n)[(n) in [0,1]] in t] in t = (n)[(n)[(n) in [0,1]] in n] in t
        new_tree[noncycle_locs[contracted_subtree]] = noncycle_locs[contracted_tree[contracted_subtree]]
        # print(1, new_tree)
        # fixed tree with heads coming from the cycle: (n) in [0,1]
        contracted_subtree = np.logical_not(contracted_subtree)
        # add the nodes to the tree (t)[(n)[(n) in [0,1]] in t] in t = (c)[(n)[(n) in [0,1]] in c] in t
        new_tree[noncycle_locs[contracted_subtree]] = cycle_locs[metanode_deps[contracted_subtree]]
        # print(2, new_tree)
        # add the old cycle to the tree; (t)[(c) in t] in t = (t)[(c) in t] in t
        new_tree[cycle_locs] = tree[cycle_locs]
        # print(3, new_tree)
        # root of the cycle; (n)[() in n] in c = () in c
        cycle_root = metanode_heads[cycle_head]
        # add the root of the cycle to the new tree; (t)[(c)[() in c] in t] = (c)[() in c]
        new_tree[cycle_locs[cycle_root]] = noncycle_locs[cycle_head]
        # print(4, new_tree)
        return new_tree


# ===============================================================
def chuliu_edmonds_one_root(scores):
    """"""
    scores = scores.astype(np.float64)
    tree = chuliu_edmonds(scores)
    roots_to_try = np.where(np.equal(tree[1:], 0))[0] + 1
    if len(roots_to_try) == 1:
        return tree

    # Look at all roots that are more likely than we would expect
    if len(roots_to_try) == 0:
        roots_to_try = np.where(scores[1:, 0] >= 1 / len(scores))[0] + 1
    # *sigh* just grab the most likely one
    if len(roots_to_try) == 0:
        roots_to_try = np.array([np.argmax(scores[1:, 0]) + 1])

    # -------------------------------------------------------------
    def set_root(scores, root):
        root_score = scores[root, 0]
        scores = np.array(scores)
        scores[1:, 0] = 0
        scores[root] = 0
        scores[root, 0] = 1
        return scores, root_score

    # -------------------------------------------------------------

    best_score, best_tree = -np.inf, None  # This is what's causing it to crash
    for root in roots_to_try:
        _scores, root_score = set_root(scores, root)
        _tree = chuliu_edmonds(_scores)
        tree_probs = _scores[np.arange(len(_scores)), _tree]
        tree_score = np.log(tree_probs).sum() + np.log(root_score) if tree_probs.all() else -np.inf
        if tree_score > best_score:
            best_score = tree_score
            best_tree = _tree
    try:
        assert best_tree is not None
    except:
        with open('debug.log', 'w') as f:
            f.write('{}: {}, {}\n'.format(tree, scores, roots_to_try))
            f.write('{}: {}, {}, {}\n'.format(_tree, _scores, tree_probs, tree_score))
        raise
    return best_tree


# ***************************************************************
def main(n=10):
    """"""

    for i in range(100):
        scores = np.random.randn(n, n)
        scores = np.exp(scores) / np.exp(scores).sum()
        scores *= (1 - np.eye(n))
        newtree = chuliu_edmonds_one_root(scores)
        cycles = tarjan(newtree)
        roots = np.where(np.equal(newtree[1:], 0))[0] + 1
        print(newtree, cycles, roots)
        assert not cycles
        assert len(roots) == 1
    return


def example1():
    w2i = defaultdict(lambda: len(w2i))
    sentence = "root Book that flight".split()
    sentence_ids = [w2i[token] for token in sentence]
    num_words = len(sentence)
    i2w = {i: w for w, i in w2i.items()}

    print(sentence)
    print(sentence_ids)
    print(num_words)

    scores = np.full([num_words, num_words], -1.)
    scores[w2i['root']][w2i['Book']] = 12.
    scores[w2i['root']][w2i['that']] = 4.
    scores[w2i['root']][w2i['flight']] = 4.
    scores[w2i['Book']][w2i['that']] = 5.
    scores[w2i['Book']][w2i['flight']] = 7.
    scores[w2i['that']][w2i['Book']] = 6.
    scores[w2i['that']][w2i['flight']] = 8.
    scores[w2i['flight']][w2i['Book']] = 5.
    scores[w2i['flight']][w2i['that']] = 7.

    print(scores)
    heads = chuliu_edmonds(scores.T)
    print("final heads:", heads)
    print("correct heads:", [0, 0, 3, 1])


def example2():
    w2i = defaultdict(lambda: len(w2i))
    sentence = "v1 v2 v3 v4 v5 v6 v7".split()
    sentence_ids = [w2i[token] for token in sentence]
    num_words = len(sentence)
    i2w = {i: w for w, i in w2i.items()}

    print(sentence)
    print(sentence_ids)
    print(num_words)

    scores = np.full([num_words, num_words], -1.)

    scores[w2i['v1']][w2i['v2']] = 9.
    scores[w2i['v1']][w2i['v5']] = 5.

    scores[w2i['v2']][w2i['v3']] = 3.
    scores[w2i['v2']][w2i['v4']] = 9.

    scores[w2i['v3']][w2i['v2']] = 7.
    scores[w2i['v3']][w2i['v6']] = 9.
    scores[w2i['v3']][w2i['v7']] = 6.

    scores[w2i['v4']][w2i['v1']] = 3.
    scores[w2i['v4']][w2i['v3']] = 8.
    scores[w2i['v4']][w2i['v6']] = 5.

    scores[w2i['v5']][w2i['v4']] = 4.

    scores[w2i['v6']][w2i['v5']] = 3.
    scores[w2i['v6']][w2i['v7']] = 4.

    scores[w2i['v7']][w2i['v3']] = 4.
    scores[w2i['v7']][w2i['v6']] = 8.

    print(scores)
    heads = chuliu_edmonds(scores.T)
    print("final heads:", heads)
    print("correct heads:", [0, 2, 6, 4, 0, 3, 5])
    print(sentence)
    print([sentence[i] for i in heads])


def example3():
    w2i = defaultdict(lambda: len(w2i))
    sentence = "v1 v2 v3 v4 v5 v6 v7".split()
    sentence_ids = [w2i[token] for token in sentence]
    num_words = len(sentence)
    i2w = {i: w for w, i in w2i.items()}

    print(sentence)
    print(sentence_ids)
    print(num_words)

    scores = np.full([num_words, num_words], -1.)

    scores[w2i['v1']][w2i['v2']] = 6.
    scores[w2i['v1']][w2i['v5']] = 5.

    scores[w2i['v2']][w2i['v3']] = 7.

    scores[w2i['v3']][w2i['v2']] = 8.
    scores[w2i['v3']][w2i['v4']] = 3.

    scores[w2i['v4']][w2i['v7']] = 6.
    scores[w2i['v4']][w2i['v6']] = 8.

    scores[w2i['v5']][w2i['v4']] = 6.

    scores[w2i['v6']][w2i['v5']] = 9.

    scores[w2i['v7']][w2i['v4']] = 2.

    print(scores)
    heads = chuliu_edmonds(scores.T)
    print("final heads:", heads)
    print("correct heads: [0 0 1 4 0 3 3]")
    print(sentence)
    print([sentence[i] for i in heads])


# ***************************************************************
if __name__ == '__main__':
    """"""
    example1()
