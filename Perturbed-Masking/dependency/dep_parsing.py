import argparse
from tqdm import tqdm
import numpy as np
import pickle
from .eisner import Eisner, decode_mst, _mst
from .standfordMST import chuliu_edmonds
from .evaluation import _evaluation
import unicodedata


def softmax(x, axis=1):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=axis)


def find_root(parse):
    # root node's head also == 0, so have to be removed
    for token in parse[1:]:
        if token.head == 0:
            return token.id
    return False


def _run_strip_accents(text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output)


def match_tokenized_to_untokenized(subwords, sentence, cls_token='[CLS]', sep_token='[SEP]',
                                   unk_token='[UNK]'):
    # subwords: [cls, sub_tok1, sub_tok2, ..., sep]
    # sentence: [w1, w2, ...]，没有拆分的sentence
    # return 与subwords一样长的list[]，内容是每个subword对应的word的序号
    token_subwords = np.zeros(len(sentence))
    sentence = [_run_strip_accents(x) for x in sentence]
    token_ids, subwords_str, current_token, current_token_normalized = [-1] * len(subwords), "", 0, None
    for i, subword in enumerate(subwords):
        # TODO 需要修改
        if subword in [cls_token, sep_token]: continue

        while current_token_normalized is None:
            current_token_normalized = sentence[current_token].lower()

        if subword.startswith(unk_token):
            unk_length = int(subword[6:])
            subwords[i] = subword[:5]
            subwords_str += current_token_normalized[len(subwords_str):len(subwords_str) + unk_length]
        else:
            subwords_str += subword[2:] if subword.startswith("##") else subword
        if not current_token_normalized.startswith(subwords_str):
            return False

        token_ids[i] = current_token
        token_subwords[current_token] += 1
        if current_token_normalized == subwords_str:
            subwords_str = ""
            current_token += 1
            current_token_normalized = None

    assert current_token_normalized is None
    while current_token < len(sentence):
        assert not sentence[current_token]
        current_token += 1
    assert current_token == len(sentence)

    return token_ids


def decoding_new(args, out=None):
    trees = []
    if out is None:
        with open(args.matrix, 'rb') as f:
            results = pickle.load(f)
    else:
        results = out
    new_results = []
    decoder = Eisner()
    root_found = 0
#  matrix 10x10, line 9(不包含sep) mapping 10
    for (line, mapping, matrix_as_list) in tqdm(results):
        orginal_line = line
        sentence = [x.form for x in line][1:] # 删除root
        # deprels.append([x.deprel for x in line])
        if args.root=='gold':
            root = find_root(line)
        else:
            root = False

        init_matrix = matrix_as_list

        # merge subwords in one row 
        merge_column_matrix = []
        for i, line in enumerate(init_matrix):
            new_row = []
            buf = []
            for j in range(0, len(line) - 1): # 不考虑结尾的sep
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
        for i, line_ in enumerate(merge_column_matrix):
            new_row = []
            buf = []
            for j in range(0, len(line_) - 1):
                buf.append(line_[j])
                if mapping[j] != mapping[j + 1]:
                    if args.subword == 'sum':
                        new_row.append(sum(buf))
                    elif args.subword == 'avg':
                        new_row.append((sum(buf) / len(buf)))
                    elif args.subword == 'first':
                        new_row.append(buf[0])
                    buf = []
            final_matrix.append(new_row)

        # transpose to the original matrix
        final_matrix = np.array(final_matrix).transpose()

        if final_matrix.shape[0] == 0:
            print('find empty matrix:',sentence)
            continue
        assert final_matrix.shape[0] == final_matrix.shape[1]

        new_results.append((orginal_line, 0, 0))

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
                final_matrix[0, 0] = 0

            # final_matrix = softmax(final_matrix)
            if hasattr(args, 'weight') and args.weight>0:
                dist_mat = np.zeros(final_matrix.shape)
                n = final_matrix.shape[0]
                for i, row in enumerate(dist_mat):
                    for j, cell in enumerate(row):
                        # dist_mat[i, j] = abs(j-i)
                        dist_mat[i, j] = 1 - abs(j - i) / n
                dist_mat = args.weight * dist_mat

                for i, row in enumerate(final_matrix):
                    for j, cell in enumerate(row):
                        final_matrix[i, j] *= dist_mat[i, j]

            final_matrix = final_matrix.transpose()  #note这边又进行了一次transpose

            # best_heads, _ = decoder.parse_proj(final_matrix)
            best_heads, _ = decode_mst(final_matrix, length=len(final_matrix), has_labels=False)
            # best_heads = _mst(final_matrix.T)
            for i, head in enumerate(best_heads):
                if head == 0 and i == root:
                    root_found += 1
            trees.append([(i, head) for i, head in enumerate(best_heads)])
        if args.decoder == 'right_chain':
            trees.append([(i, 0) if i == final_matrix.shape[0]-1 else (i, i + 1) for i in range(0, final_matrix.shape[0])])

        if args.decoder == 'gold':
            trees.append([(i, x.head) for i, x in enumerate(orginal_line)])

        if args.decoder == 'left_chain':
            trees.append([(0, 0) if i == 0 else (i, i - 1) for i in range(0, final_matrix.shape[0])])
            # trees.append([(i, i + 1) for i in range(0, final_matrix.shape[0])])
    # 确认一下有cls的没： 有的，这里应该是故意保留cls，来充当root的位置
    # 返回的trees是
    #   [ [(0, head), (1, head), ...],
    #     [(0, head), (1, head), ...],
    #     ...
    #   ]
    return trees, new_results


#TODO 需要重写decoding, 这样只需要得到那个tree就好
def decoding(args):
    trees = []
    deprels = []
    with open(args.matrix, 'rb') as f:
        results = pickle.load(f)
    new_results = []
    decoder = Eisner()
    root_found = 0

    for (line, tokenized_text, matrix_as_list) in tqdm(results):
        orginal_line = line
        sentence = [x.form for x in line][1:] # 不考虑结尾的sep
        deprels.append([x.deprel for x in line])
        root = find_root(line)

        mapping = match_tokenized_to_untokenized(tokenized_text, sentence)

        init_matrix = matrix_as_list

        # merge subwords in one row
        merge_column_matrix = []
        for i, line in enumerate(init_matrix):
            new_row = []
            buf = []
            for j in range(0, len(line) - 1): # 不考虑结尾的sep
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
                    if args.subword == 'sum':
                        new_row.append(sum(buf))
                    elif args.subword == 'avg':
                        new_row.append((sum(buf) / len(buf)))
                    elif args.subword == 'first':
                        new_row.append(buf[0])
                    buf = []
            final_matrix.append(new_row)

        # transpose to the original matrix
        final_matrix = np.array(final_matrix).transpose()

        if final_matrix.shape[0] == 0:
            print('find empty matrix:',sentence)
            continue
        assert final_matrix.shape[0] == final_matrix.shape[1]

        new_results.append((orginal_line, tokenized_text, matrix_as_list))

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
                final_matrix[0, 0] = 0

            final_matrix = softmax(final_matrix)
            final_matrix = final_matrix.transpose()

            best_heads, _ = decoder.parse_proj(final_matrix)
            for i, head in enumerate(best_heads):
                if head == 0 and i == root:
                    root_found += 1
            trees.append([(i, head) for i, head in enumerate(best_heads)])
        if args.decoder == 'right_chain':
            trees.append([(root, 0) if i == root else (i, i + 1) for i in range(0, final_matrix.shape[0])])
            # trees.append([(i, i + 1) for i in range(0, final_matrix.shape[0])])
    # 确认一下有cls的没： 有的
    # 返回的trees是
    #   [ [(word_idx, head), (word_idx, head), ...],
    #     [(word_idx, head), (word_idx, head), ...],
    #     ...
    #   ]
    return trees, new_results, deprels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data args
    parser.add_argument('--matrix', default='../results/dependency/bert-dist-last_layer.pkl')
    # parser.add_argument('--matrix', default='./results/WSJ10/bert-base-uncased-False-diff.pkl')
    # parser.add_argument('--matrix', default='./results/WSJ10/bert-base-uncased-False-dist-12.pkl')

    # Decoding args
    parser.add_argument('--decoder', default='eisner')
    parser.add_argument('--root', default='gold', help='gold or cls')
    parser.add_argument('--subword', default='first')

    args = parser.parse_args()
    print(args)
    trees, results, deprels = decoding(args)
    _evaluation(trees, results)
    # distance_analysis(trees, results)
    # distance_tag_analysis(trees, results, deprels)
    # tag_evaluation(trees, results, deprels)
    # no_punc_evaluation(trees, results, deprels)