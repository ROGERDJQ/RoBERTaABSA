from transformers import BertModel, BertTokenizer
import torch
import pickle
import argparse
from tqdm import tqdm
import numpy as np
import os
from .subword_script import match_tokenized_to_untokenized
from .data_ptb import Corpus


def get_all_subword_id(mapping, idx):
    current_id = mapping[idx]
    id_for_all_subwords = [tmp_id for tmp_id, v in enumerate(mapping) if v == current_id]
    return id_for_all_subwords


def get_con_matrix(args, model, tokenizer):
    corpus = Corpus(args.dataset, args.data_split)

    mask_id = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]

    model.eval()

    LAYER = int(args.layers)
    LAYER += 1  # also consider embedding layer
    out = [[] for i in range(LAYER)]
    for sents, tree2list, nltk_tree in tqdm(zip(corpus.sens, corpus.trees, corpus.nltk_trees), total=len(corpus.sens)):

        sentence = ' '.join(sents)
        tokenized_text = tokenizer.tokenize(sentence)
        tokenized_text.insert(0, '[CLS]')
        tokenized_text.append('[SEP]')
        # Convert token to vocabulary indices
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

        mapping = match_tokenized_to_untokenized(tokenized_text, sents)

        # 1. Generate mask indices
        all_layers_matrix_as_list = [[] for i in range(LAYER)]
        for i in range(0, len(tokenized_text)):
            id_for_all_i_tokens = get_all_subword_id(mapping, i)
            tmp_indexed_tokens = list(indexed_tokens)
            for tmp_id in id_for_all_i_tokens:
                if mapping[tmp_id] != -1:  # both CLS and SEP use -1 as id e.g., [-1, 0, 1, 2, ..., -1]
                    tmp_indexed_tokens[tmp_id] = mask_id
            one_batch = [list(tmp_indexed_tokens) for _ in range(0, len(tokenized_text))]
            for j in range(0, len(tokenized_text)):
                id_for_all_j_tokens = get_all_subword_id(mapping, j)
                for tmp_id in id_for_all_j_tokens:
                    if mapping[tmp_id] != -1:
                        one_batch[j][tmp_id] = mask_id

            # 2. Convert one batch to PyTorch tensors
            tokens_tensor = torch.tensor(one_batch)
            segments_tensor = torch.tensor([[0 for _ in one_sent] for one_sent in one_batch])
            if args.cuda:
                tokens_tensor = tokens_tensor.to('cuda')
                segments_tensor = segments_tensor.to('cuda')
                model.to('cuda')

            # 3. get all hidden states for one batch
            with torch.no_grad():
                model_outputs = model(tokens_tensor, segments_tensor)
                all_layers = model_outputs[-1]  # 12 layers + embedding layer

            # 4. get hidden states for word_i in one batch
            for k, layer in enumerate(all_layers):
                if args.cuda:
                    hidden_states_for_token_i = layer[:, i, :].cpu().numpy()
                else:
                    hidden_states_for_token_i = layer[:, i, :].numpy()
                all_layers_matrix_as_list[k].append(hidden_states_for_token_i)

        for k, one_layer_matrix in enumerate(all_layers_matrix_as_list):
            init_matrix = np.zeros((len(tokenized_text), len(tokenized_text)))
            for i, hidden_states in enumerate(one_layer_matrix):
                base_state = hidden_states[i]
                for j, state in enumerate(hidden_states):
                    if args.metric == 'dist':
                        init_matrix[i][j] = np.linalg.norm(base_state - state)
                    if args.metric == 'cos':
                        init_matrix[i][j] = np.dot(base_state, state) / (
                                np.linalg.norm(base_state) * np.linalg.norm(state))
            out[k].append((sents, tokenized_text, init_matrix, tree2list, nltk_tree))

    for k, one_layer_out in enumerate(out):
        k_output = args.output_file.format(args.model_type, args.metric, args.data_split, str(k))
        with open(k_output, 'wb') as fout:
            pickle.dump(out[k], fout)
            fout.close()

if __name__ == '__main__':
    MODEL_CLASSES = {
        'bert': (BertModel, BertTokenizer, 'bert-base-uncased'),
    }
    parser = argparse.ArgumentParser()

    # Model args
    parser.add_argument("--model_type", default='bert', type=str)
    parser.add_argument('--layers', default='12')

    # Data args
    parser.add_argument('--data_split', default='WSJ23')
    parser.add_argument('--dataset', default='constituency/data/WSJ/')
    parser.add_argument('--output_dir', default='./results/')

    # Matrix args
    parser.add_argument('--metric', default='dist')
    parser.add_argument('--probe', default='constituency', help="dependency, constituency, discourse")

    # Cuda
    parser.add_argument('--cuda', action='store_true')

    args = parser.parse_args()

    model_class, tokenizer_class, pretrained_weights = MODEL_CLASSES[args.model_type]

    args.output_dir = args.output_dir + args.probe + '/'
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    data_split = args.dataset.split('/')[-1].split('.')[0]
    args.output_file = args.output_dir + '/{}-{}-{}-{}.pkl'

    model = model_class.from_pretrained(pretrained_weights, output_hidden_states=True)
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights, do_lower_case=True)

    print(args)
    get_con_matrix(args, model, tokenizer)
    # with open('./results/WSJ23/bert-base-uncased-False-dist-12.pkl', 'rb') as f:
    #     results = pickle.load(f)