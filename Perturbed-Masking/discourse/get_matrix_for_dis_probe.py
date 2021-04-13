from transformers import BertModel, BertTokenizer
import torch
import pickle
import argparse
from tqdm import tqdm
import numpy as np
from .data_scidtb import Corpus
import os


def span_index(span):
    tag = False
    for i, index in enumerate(span):
        if index == 1 and tag == False:
            start_idx = i
            tag = True
        if index == 0 and tag:
            end_idx = i
            tag = False
    if start_idx >= 0 and end_idx >= 0:
        return start_idx, end_idx
    else:
        return False


def get_dis_matrix(args, model, tokenizer):
    corpus = Corpus(args.dataset, tokenizer)

    model.eval()

    out = []
    for doc_id, (batch, batch_pos, tree) in tqdm(enumerate(corpus.scidtb)):
        matrix_as_list = []
        for i, one_batch in enumerate(batch):
            # edu_i 's position: start and end index
            mask_a, mask_b = span_index(batch_pos[i])

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
                last_layer = model_outputs[0]

            # 4. send to cpu
            if args.cuda:
                hidden_states_for_span_i = last_layer[:, mask_a:mask_b, :].cpu().numpy()
            else:
                hidden_states_for_span_i = last_layer[:, mask_a:mask_b, :].numpy()

            span_representations = []
            for span_i in hidden_states_for_span_i:
                span_representations.append(span_i.mean(axis=0))
            matrix_as_list.append(span_representations)

        if args.metric == 'dist':
            init_matrix = np.zeros((len(batch), len(batch)))
            for i, hidden_states in enumerate(matrix_as_list):
                base_state = hidden_states[i]
                for j, state in enumerate(hidden_states):
                    init_matrix[i][j] = np.linalg.norm(base_state - state)
            out.append((tree, init_matrix))

    with open(args.output_file.format(args.model_type, args.metric, args.data_split, 'last'), 'wb') as fout:
        pickle.dump(out, fout)


if __name__ == '__main__':
    MODEL_CLASSES = {
        'bert': (BertModel, BertTokenizer, 'bert-base-uncased'),
    }
    parser = argparse.ArgumentParser()

    # Model args
    parser.add_argument("--model_type", default='bert', type=str)
    parser.add_argument('--layers', default='12')

    # Data args
    parser.add_argument('--data_split', default='SciDTB')
    parser.add_argument('--dataset', default='./SciDTB/test/gold/')
    parser.add_argument('--output_dir', default='./results/')

    parser.add_argument('--metric', default='dist', help='metrics for impact calculation, support [dist, cos] so far')
    parser.add_argument('--cuda', action='store_true', help='invoke to use gpu')

    parser.add_argument('--probe', default='discourse', help="dependency, constituency, discourse")

    args = parser.parse_args()

    model_class, tokenizer_class, pretrained_weights = MODEL_CLASSES[args.model_type]

    args.output_dir = args.output_dir + args.probe + '/'
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.output_file = args.output_dir + '/{}-{}-{}-{}.pkl'

    model = model_class.from_pretrained(pretrained_weights, output_hidden_states=True)
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights, do_lower_case=True)

    print(args)
    get_dis_matrix(args, model, tokenizer)