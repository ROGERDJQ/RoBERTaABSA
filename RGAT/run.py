# coding=utf-8
import argparse
import logging

import os
import random
import warnings

warnings.filterwarnings("ignore")
if "p" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["p"]


import numpy as np
import torch
from transformers import (
    BertConfig,
    BertForTokenClassification,
    BertTokenizer,
    RobertaTokenizer,
)
from torch.utils.data import DataLoader

from datasets import load_datasets_and_vocabs
from model import (
    Aspect_Text_GAT_ours,
    Pure_Bert,
    Aspect_Bert_GAT,
    Aspect_Text_GAT_only,
    Aspect_Roberta_GAT,
)
from trainer import train
import fitlog

fitlog.debug()
fitlog.set_log_dir("logs")
fitlog.set_rng_seed()
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--dataset_name", type=str, help="Choose absa dataset."
    )
    parser.add_argument("--refresh", type=int, default=0, help="Generate data again")

    # Model parameters
    parser.add_argument(
        "--glove_dir",
        type=str,
        help="Directory storing glove embeddings",
    )
    parser.add_argument("--highway", action="store_true", help="Use highway embed.")
    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="Number of layers of bilstm or highway or elmo.",
    )
    parser.add_argument("--max_hop", type=int, default=4, help="max number of hops")
    parser.add_argument(
        "--num_heads", type=int, default=6, help="Number of heads for gat."
    )
    parser.add_argument(
        "--dropout", type=float, default=0.7, help="Dropout rate for embedding."
    )
    parser.add_argument(
        "--num_gcn_layers", type=int, default=1, help="Number of GCN layers."
    )
    parser.add_argument(
        "--gcn_mem_dim", type=int, default=300, help="Dimension of the W in GCN."
    )
    parser.add_argument(
        "--gcn_dropout", type=float, default=0.2, help="Dropout rate for GCN."
    )
    # GAT
    parser.add_argument(
        "--gat_attention_type",
        type=str,
        choices=["linear", "dotprod", "gcn"],
        default="dotprod",
        help="The attention used for gat",
    )

    parser.add_argument(
        "--embedding_type",
        type=str,
        default="glove",
        choices=["glove", "bert", "roberta"],
    )
    parser.add_argument(
        "--embedding_dim", type=int, default=300, help="Dimension of glove embeddings"
    )
    parser.add_argument(
        "--dep_relation_embed_dim",
        type=int,
        default=300,
        help="Dimension for dependency relation embeddings.",
    )

    parser.add_argument(
        "--hidden_size",
        type=int,
        default=300,
        help="Hidden size of bilstm, in early stage.",
    )
    parser.add_argument(
        "--final_hidden_size",
        type=int,
        default=300,
        help="Hidden size of bilstm, in early stage.",
    )
    parser.add_argument(
        "--num_mlps", type=int, default=2, help="Number of mlps in the last of model."
    )

    # Training parameters
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-3,
        type=float,
        help="The initial learning rate for Adam.",
    )

    parser.add_argument(
        "--num_train_epochs",
        default=25,
        type=int,
        help="Total number of training epochs to perform.",
    )

    args = parser.parse_args()
    if args.dataset_name.endswith("/"):
        args.dataset_name = args.dataset_name[:-1]
    fitlog.add_hyper(args)

    if "/" in args.dataset_name:
        data = os.path.basename(args.dataset_name)
        output_dir = f"data/{data}"
    else:
        output_dir = f"data/{args.dataset_name}"

    args.output_dir = output_dir

    args.lower = 1
    args.logging_steps = 30
    args.max_steps = -1
    args.max_grad_norm = 10
    args.adam_epsilon = 1e-8
    args.weight_decay = 0
    args.gradient_accumulation_steps = 1
    args.per_gpu_train_batch_size = args.batch_size
    args.per_gpu_eval_batch_size = args.batch_size * 2
    args.add_non_connect = 1
    args.multi_hop = True
    args.num_classes = 3
    args.cuda_id = "0"
    args.bert_model_dir = "roberta-en"
    args.pure_bert = False
    args.gat_our = True
    args.gat_roberta = False
    args.gat = False
    args.gat_bert = False

    return args


def check_args(args):
    """
    eliminate confilct situations

    """
    logger.info(vars(args))


def main():
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Parse args
    args = parse_args()
    if args.dataset_name.endswith("/"):
        args.dataset_name = args.dataset_name[:-1]
    dataset_name = args.dataset_name
    # 形如 ~/rgat/bert/11/Restaurants
    if "/" in dataset_name:
        pre_model_name, layer, dataset = dataset_name.split("/")[-3:]
    else:
        pre_model_name, dataset = "None", dataset_name
        layer = "-1"
    fitlog.add_hyper(value=pre_model_name, name="model_name")
    fitlog.add_hyper(value=dataset, name="dataset")
    fitlog.add_hyper(value=layer, name="pre_layer")
    fitlog.add_hyper(value="RGAT", name="model")

    # if 'Laptop' in args.dataset_name:
    #     assert args.lower == 0
    check_args(args)

    # Setup CUDA, GPU training
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    logger.info("Device is %s", args.device)

    # Bert, load pretrained model and tokenizer, check if neccesary to put bert here
    if args.embedding_type == "bert":
        tokenizer = BertTokenizer.from_pretrained(args.bert_model_dir)
        args.tokenizer = tokenizer
    elif args.embedding_type == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained(args.bert_model_dir)
        args.tokenizer = tokenizer

    # Load datasets and vocabs
    (
        train_dataset,
        test_dataset,
        word_vocab,
        dep_tag_vocab,
        pos_tag_vocab,
    ) = load_datasets_and_vocabs(args)

    # Build Model
    # model = Aspect_Text_Multi_Syntax_Encoding(args, dep_tag_vocab['len'], pos_tag_vocab['len'])
    if args.pure_bert:
        model = Pure_Bert(args)
    elif args.gat_roberta:
        model = Aspect_Roberta_GAT(args, dep_tag_vocab["len"], pos_tag_vocab["len"])
    elif args.gat_bert:
        model = Aspect_Bert_GAT(
            args, dep_tag_vocab["len"], pos_tag_vocab["len"]
        )  # R-GAT + Bert
    elif args.gat_our:
        model = Aspect_Text_GAT_ours(
            args, dep_tag_vocab["len"], pos_tag_vocab["len"]
        )  # R-GAT with reshaped tree
    else:
        model = Aspect_Text_GAT_only(
            args, dep_tag_vocab["len"], pos_tag_vocab["len"]
        )  # original GAT with reshaped tree

    model.to(args.device)
    # Train
    _, _, all_eval_results = train(args, train_dataset, model, test_dataset)

    print("\n\nBest Results:")
    if len(all_eval_results):
        best_eval_result = max(all_eval_results, key=lambda x: x["acc"])
        step = [
            i for i, result in enumerate(all_eval_results) if result == best_eval_result
        ][0]
        logger.info("Achieve at step {}/{}".format(step, len(all_eval_results)))
        for key in sorted(best_eval_result.keys()):
            logger.info("  %s = %s", key, str(best_eval_result[key]))
        # fitlog.add_best_metric(value=best_eval_result['acc'], name='acc')
        # fitlog.add_best_metric(value=best_eval_result['f1'], name='f1')
    fitlog.finish()


if __name__ == "__main__":
    main()
