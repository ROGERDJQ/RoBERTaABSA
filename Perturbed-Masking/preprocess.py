import os
import argparse

if "p" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["p"]
import warnings

warnings.filterwarnings("ignore")
from utils import ConllUDataset, ResLoader
from transformers import BertModel, BertTokenizer, RobertaTokenizer, RobertaModel
from dependency import get_dep_matrix, get_dep_matrix_new
from constituency import get_con_matrix
from discourse import get_dis_matrix


if __name__ == "__main__":
    MODEL_CLASSES = {
        "bert": (BertModel, BertTokenizer, "bert-base-uncased"),
        "roberta": (RobertaModel, RobertaTokenizer, "roberta-base"),
    }
    parser = argparse.ArgumentParser()

    # Model args
    parser.add_argument("--model_type", default="bert", type=str)
    parser.add_argument("--model_path", default="", type=str)
    parser.add_argument("--layers", default="12")

    # Data args
    # data_split: PUD, Laptop-test
    parser.add_argument("--data_split", default="PUD")
    parser.add_argument(
        "--dataset", default="./discourse/SciDTB/test/gold/", required=True
    )
    parser.add_argument("--output_dir", default="./results/")

    parser.add_argument(
        "--metric",
        default="dist",
        help="metrics for impact calculation, support [dist, cos] so far",
    )
    parser.add_argument("--cuda", action="store_true", help="invoke to use gpu")

    parser.add_argument(
        "--probe",
        required=True,
        choices=["dependency", "constituency", "discourse", "new_dep"],
    )

    args = parser.parse_args()

    model_class, tokenizer_class, pretrained_weights = MODEL_CLASSES[args.model_type]
    if args.model_path:
        pretrained_weights = args.model_path

    args.output_dir = os.path.join(args.output_dir, args.probe)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.output_file = args.output_dir + "/{}-{}-{}-{}.pkl"

    model = model_class.from_pretrained(pretrained_weights, output_hidden_states=True)
    tokenizer = tokenizer_class.from_pretrained(
        MODEL_CLASSES[args.model_type][2]
    )  # tokenizer都不会改变的

    print(args)

    if args.probe == "dependency":
        dataset = ConllUDataset(args.dataset)
        get_dep_matrix(args, model, tokenizer, dataset)
    elif args.probe == "new_dep":
        get_dep_matrix_new(args, model, tokenizer, dataset)
    elif args.probe == "constituency":
        get_con_matrix(args, model, tokenizer)
    elif args.probe == "discourse":
        get_dis_matrix(args, model, tokenizer)
