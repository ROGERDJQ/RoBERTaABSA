from collections import defaultdict


def undirected_standard(gold):
    undirected = [(head, dependent) for (dependent, head) in gold]
    return gold+undirected


def evaluation(trees, gold_trees):
    uas_count, total_relations = 0., 0.
    uuas_count = 0.
    for tree, gold_tree in zip(trees, gold_trees):
        directed_gold_edges = [(edu['id'], edu['parent']) for edu in gold_tree][1:]
        undirected_gold_edges = undirected_standard(directed_gold_edges)

        identical = set(directed_gold_edges) & set(tree)
        undirected_identical = set(undirected_gold_edges) & set(tree)

        total_relations += len(directed_gold_edges)
        uas_count += len(identical)
        uuas_count += len(undirected_identical)
    uas = uas_count / total_relations
    uuas = uuas_count / total_relations
    print("UAS, UUAS:", uas, uuas)
    print("correct and total arcs", uas_count, total_relations)
    print('remove root acc:', (uas_count-152)/total_relations)
    return uas


def find_root(parse):
    for edu in parse:
        if edu['parent'] == 0:
            return edu['id']
    return False


def distance_evaluation(trees, gold_trees):
    n_correct = defaultdict(lambda :0)
    n_incorrect = defaultdict(lambda :0)
    for tree, gold_tree in zip(trees, gold_trees):
        root = find_root(gold_tree)
        directed_gold_edges = [(edu['id'], edu['parent']) for edu in gold_tree]
        gold_distance = [abs(x[1]-x[0])-1 for x in directed_gold_edges]
        for i, (p, g, d) in enumerate(zip(tree, directed_gold_edges, gold_distance)):
            if i == 0:
                continue
            # if i == root:
            #     continue
            # remove the node that point to the root
            if g[1] == 0:
                continue
            is_correct = (p == g)
            if is_correct:
                n_correct[d] += 1
            else:
                n_incorrect[d] += 1
    long_correct, long_incorrect = 0, 0
    for k in sorted(n_incorrect.keys()):
        if int(k) < 6:
            print('{}\t{:.3f}\t{}'.format(k, 100 * n_correct[k] / float(n_correct[k] + n_incorrect[k]),
                                      n_correct[k] + n_incorrect[k]))
        long_correct +=  n_correct[k]
        long_incorrect += n_incorrect[k]
    print(long_correct / (long_correct+long_incorrect))
