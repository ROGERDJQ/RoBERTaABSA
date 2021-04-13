import nltk
import argparse
import pickle
import numpy as np
from .decoder import mart, right_branching, left_branching
import re
word_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
             'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
             'WDT', 'WP', 'WP$', 'WRB']
from collections import Counter
from .con_eval import corpus_stats_labeled, corpus_average_depth
from .subword_script import match_tokenized_to_untokenized


def get_brackets(tree, idx=0):
    brackets = set()
    if isinstance(tree, list) or isinstance(tree, nltk.Tree):
        for node in tree:
            node_brac, next_idx = get_brackets(node, idx)
            if next_idx - idx > 1:
                brackets.add((idx, next_idx))
                brackets.update(node_brac)
            idx = next_idx
        return brackets, idx
    else:
        return brackets, idx + 1


def MRG(tr):
    if isinstance(tr, str):
        #return '(' + tr + ')'
        return tr + ' '
    else:
        s = '( '
        for subtr in tr:
            s += MRG(subtr)
        s += ') '
        return s


def MRG_labeled(tr):
    if isinstance(tr, nltk.Tree):
        if tr.label() in word_tags:
            return tr.leaves()[0] + ' '
        else:
            s = '(%s ' % (re.split(r'[-=]', tr.label())[0])
            for subtr in tr:
                s += MRG_labeled(subtr)
            s += ') '
            return s
    else:
        return ''


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def decoding(args):
    trees = []
    new_results = []
    with open(args.matrix, 'rb') as f:
        results = pickle.load(f)

    for (sen, tokenized_text, init_matrix, tree2list, nltk_tree) in results:
        mapping = match_tokenized_to_untokenized(tokenized_text, sen)
        # merge subwords in one row
        merge_column_matrix = []
        for i, line in enumerate(init_matrix):
            new_row = []
            buf = []
            for j in range(0, len(line) - 1):
                buf.append(line[j])
                if mapping[j] != mapping[j + 1]:
                    new_row.append(buf[0])
                    buf = []
            merge_column_matrix.append(new_row)

        # merge subwords in multi rows
        # transpose the matrix so we can work with row instead of multiple rows
        merge_column_matrix = np.array(merge_column_matrix).transpose()
        merge_column_matrix = merge_column_matrix.tolist()
        final_matrix = []
        for i, line in enumerate(merge_column_matrix):
            new_row = []
            buf = []
            for j in range(0, len(line) - 1):
                buf.append(line[j])
                if mapping[j] != mapping[j + 1]:
                    if args.subword == 'max':
                        new_row.append(max(buf))
                    elif args.subword == 'avg':
                        new_row.append((sum(buf) / len(buf)))
                    elif args.subword == 'first':
                        new_row.append(buf[0])
                    buf = []
            final_matrix.append(new_row)

        # transpose to the original matrix
        final_matrix = np.array(final_matrix).transpose()

        # filter some empty matrix (only one word)
        if final_matrix.shape[0] == 0:
            print(final_matrix.shape)
            continue
        assert final_matrix.shape[0] == final_matrix.shape[1]
        new_results.append((sen, tokenized_text, init_matrix, tree2list, nltk_tree))
        final_matrix = final_matrix[1:, 1:]

        final_matrix = softmax(final_matrix)

        np.fill_diagonal(final_matrix, 0.)

        final_matrix = 1. - final_matrix
        np.fill_diagonal(final_matrix, 0.)

        if args.decoder == 'mart':
            parse_tree = mart(final_matrix, sen)
            trees.append(parse_tree)

        if args.decoder == 'right_branching':
            trees.append(right_branching(sen))

        if args.decoder == 'left_branching':
            trees.append(left_branching(sen))

    return trees, new_results


def constituent_evaluation(trees, results):
    prec_list = []
    reca_list = []
    f1_list = []

    corpus_sys = {}
    corpus_ref = {}

    nsens = 0

    for tree, result in zip(trees, results):
        nsens += 1
        sen, tokenized_text, init_matrix, tree2list, nltk_tree = result

        corpus_sys[nsens] = MRG(tree)
        corpus_ref[nsens] = MRG_labeled(nltk_tree)

        model_out, _ = get_brackets(tree)
        std_out, _ = get_brackets(tree2list)
        overlap = model_out.intersection(std_out)

        prec = float(len(overlap)) / (len(model_out) + 1e-8)
        reca = float(len(overlap)) / (len(std_out) + 1e-8)
        if len(std_out) == 0:
            reca = 1.
            if len(model_out) == 0:
                prec = 1.
        f1 = 2 * prec * reca / (prec + reca + 1e-8)
        prec_list.append(prec)
        reca_list.append(reca)
        f1_list.append(f1)

    prec_list, reca_list, f1_list \
        = np.array(prec_list).reshape((-1, 1)), np.array(reca_list).reshape((-1, 1)), np.array(
        f1_list).reshape((-1, 1))
    print('-' * 80)
    np.set_printoptions(precision=4)
    print('Mean Prec:', prec_list.mean(axis=0),
          ', Mean Reca:', reca_list.mean(axis=0),
          ', Mean F1:', f1_list.mean(axis=0))
    # print('Number of sentence: %i' % len(trees))

    # correct, total = corpus_stats_labeled(corpus_sys, corpus_ref)
    # for key in correct.keys():
        # if total[key] > 500:
            # print('{}: {} / {} , accuracy: {}'.format(key, correct[key], total[key], correct[key]/total[key]))
    # print('ADJP:', correct['ADJP']/total['ADJP'])
    # print('NP:', correct['NP']/total['NP'])
    # print('PP:', correct['PP']/total['PP'])
    # print('INTJ:', correct['INTJ']/total['INTJ'])
    # print('average depth:', corpus_average_depth(corpus_sys))
    # return correct, total, corpus_average_depth(corpus_sys)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data args
    parser.add_argument('--matrix', default='../results/constituency/bert-dist-WSJ23-12.pkl')

    # Decoding args
    parser.add_argument('--decoder', default='mart')
    parser.add_argument('--subword', default='avg')

    args = parser.parse_args()
    trees, results = decoding(args)
    constituent_evaluation(trees, results)

