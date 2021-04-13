import collections


def undirected_standard(gold):
    undirected = [(head, dependent) for (dependent, head) in gold]
    return gold+undirected


def ned_standard(gold):
    undirected = [(head, dependent) for (dependent, head) in gold]
    tree = collections.defaultdict(lambda: -1)
    for (dependent, head) in gold:
        tree[dependent] = head
    ned = []
    for (dependent, head) in gold:
        grandparent = tree[head]
        if grandparent != -1:
            ned.append((dependent, grandparent))

    return gold + undirected + ned


def _evaluation(trees, results):
    uas_count, total_relations = 0., 0.
    uuas_count = 0.
    ned_count = 0.
    for tree, result in zip(trees, results):
        line, tokenized_text, matrix_as_list = result
        directed_gold_edges = [(x.id, x.head) for x in line][1:]  # remove root
        undirected_gold_edges = undirected_standard(directed_gold_edges)
        ned_gold_edges = ned_standard(directed_gold_edges)

        identical = set(directed_gold_edges) & set(tree)
        undirected_identical = set(undirected_gold_edges) & set(tree)
        ned_identical = set(ned_gold_edges) & set(tree)

        total_relations += len(directed_gold_edges)
        uas_count += len(identical)
        uuas_count += len(undirected_identical)
        ned_count += len(ned_identical)
    uas = uas_count / total_relations
    uuas = uuas_count / total_relations
    ned = ned_count / total_relations
    print("UAS, UUAS, NED:", uas, uuas, ned)
    print("correct and total arcs", uas_count, total_relations)
    return uas, uuas, ned

