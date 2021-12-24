import argparse
import os
import sys
import warnings

import fitlog
import torch
import torch.nn.functional as F
from torch import nn, optim
from tqdm import tqdm
warnings.filterwarnings("ignore")
from fastNLP import (AccuracyMetric, BucketSampler, ClassifyFPreRecMetric,
                     ConstantTokenNumSampler, CrossEntropyLoss, DataSetIter,
                     FitlogCallback, LossBase, RandomSampler,
                     SequentialSampler, SortedSampler, Trainer, WarmupCallback,
                     cache_results)
from fastNLP.core.utils import (_get_model_device, _move_dict_value_to_device,
                                _move_model_to_device)
from fastNLP.embeddings import BertWordPieceEncoder, RobertaWordPieceEncoder
from transformers import XLMRobertaModel, XLNetModel

from pipe import DataPipe

# fitlog.debug()
root_fp = r"/your/work/space/RoBERTaABSA/Train"
os.makedirs(f"{root_fp}/FT_logs", exist_ok=True)


fitlog.set_log_dir(f"{root_fp}/FT_logs")
fitlog.set_rng_seed()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    default="Laptop",
    choices=[
        "Restaurants",
        "Laptop",
        "Tweets",
        "fr",
        "sp",
        "dutch",
    ],
)
parser.add_argument(
    "--data_dir",
    type=str,
    help="dataset dir, should concat with dataset arguement",
)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument(
    "--model_name",
    type=str,
    default="roberta-en",
    choices=[
        "bert-en-base-uncased",
        "roberta-en",
        "roberta-en-large",
        "xlmroberta-xlm-roberta-base",
        "bert-multi-base-cased",
        "xlnet-xlnet-base-cased",
    ],
)
parser.add_argument("--save_embed", default=1, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--n_epochs", default=20, type=int)
parser.add_argument("--pool", default="max")
parser.add_argument("--dropout", default=0.5,type=float)
parser.add_argument("--warmup",default=0.01,type=float)

args = parser.parse_args()

args.data_dir = r"/your/work/space/RoBERTaABSA/Dataset"

fitlog.add_hyper(args)
print(args)



model_type = args.model_name.split("-")[0]
if model_type == "bert":
    mask = "[UNK]"
elif model_type == "roberta":
    mask = "<mask>"
elif model_type == "xlnet":
    mask = "<mask>"
elif model_type == "xlmroberta":
    mask = "<mask>"


@cache_results(
    f"{root_fp}/caches/data_{args.dataset}_{mask}_{args.model_name}.pkl",
    _refresh=False,
)
def get_data():
    data_bundle = DataPipe(model_name=args.model_name, mask=mask).process_from_file(
        os.path.join(args.data_dir, args.dataset)
    )
    return data_bundle


data_bundle = get_data()

print(data_bundle)

if args.model_name.split("-")[0] in ("bert", "roberta", "xlnet", "xlmroberta"):
    model_type, model_name = (
        args.model_name[: args.model_name.index("-")],
        args.model_name[args.model_name.index("-") + 1 :],
    )

if model_type == "roberta":
    embed = RobertaWordPieceEncoder(model_dir_or_name=model_name, requires_grad=True)
elif model_type == "bert":
    embed = BertWordPieceEncoder(model_dir_or_name=model_name, requires_grad=True)
elif model_type == "xlnet":
    embed = XLNetModel.from_pretrained(pretrained_model_name_or_path=model_name)
elif model_type == "xlmroberta":
    embed = XLMRobertaModel.from_pretrained(pretrained_model_name_or_path=model_name)


class AspectModel(nn.Module):
    def __init__(self, embed, dropout, num_classes, pool="max"):
        super().__init__()
        assert pool in ("max", "mean")
        self.embed = embed
        self.embed_dropout = nn.Dropout(dropout)
        if hasattr(embed, "embedding_dim"):
            embed_size = embed.embedding_dim
        else:
            embed_size = embed.config.hidden_size
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embed_size, num_classes),
        )
        self.pool = pool

    def forward(self, tokens, aspect_mask):
        """

        :param tokens:
        :param aspect_mask: bsz x max_len, 1 for aspect
        :return:
        """
        if isinstance(self.embed, BertWordPieceEncoder):
            tokens = self.embed(tokens, None)  # bsz x max_len x hidden_size
        else:
            tokens = self.embed(
                tokens, token_type_ids=None
            )  # bsz x max_len x hidden_size

        if isinstance(tokens, tuple):
            tokens = tokens[0]

        tokens = self.embed_dropout(tokens)

        aspect_mask = aspect_mask.eq(1)
        if self.pool == "mean":
            tokens = tokens.masked_fill(aspect_mask.unsqueeze(-1).eq(0), 0)
            tokens = tokens.sum(dim=1)
            preds = tokens / aspect_mask.sum(dim=1, keepdims=True).float()
        elif self.pool == "max":
            aspect_mask = aspect_mask.unsqueeze(-1).eq(0)  # bsz x max_len x 1
            tokens = tokens.masked_fill(aspect_mask, -10000.0)
            preds, _ = tokens.max(dim=1)
        preds = self.ffn(preds)
        return {"pred": preds}


model = AspectModel(
    embed,
    dropout=args.dropout,
    num_classes=len(data_bundle.get_vocab("target")) - 1,
    pool=args.pool,
)

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [
            p
            for n, p in model.named_parameters()
            if not any(nd in n for nd in no_decay)
        ],
        "weight_decay": 1e-2,
    },
    {
        "params": [
            p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
        ],
        "weight_decay": 0.0,
    },
]
optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.lr)

callbacks = []
callbacks.append(WarmupCallback(args.warmup, "constant"))
callbacks.append(FitlogCallback())


class SmoothLoss(LossBase):
    def __init__(self, smooth_eps=0):
        super().__init__()
        self.smooth_eps = smooth_eps

    def get_loss(self, pred, target):
        """

        :param pred: bsz x 3
        :param target: bsz,
        :return:
        """
        n_class = pred.size(1)
        smooth_pos = target.eq(n_class)
        target = target.masked_fill(smooth_pos, 0)
        target_matrix = torch.full_like(
            pred, fill_value=self.smooth_eps / (n_class - 1)
        )
        target_matrix = target_matrix.scatter(
            dim=1, index=target.unsqueeze(1), value=1 - self.smooth_eps
        )
        target_matrix = target_matrix.masked_fill(
            smooth_pos.unsqueeze(1), 1.0 / n_class
        )

        pred = F.log_softmax(pred, dim=-1)
        loss = -(pred * target_matrix).sum(dim=-1).mean()
        return loss


tr_data = DataSetIter(
    data_bundle.get_dataset("train"),
    num_workers=2,
    batch_sampler=ConstantTokenNumSampler(
        data_bundle.get_dataset("train").get_field("seq_len").content,
        max_token=2000,
        num_bucket=10,
    ),
)


trainer = Trainer(
    tr_data,
    model,
    optimizer=optimizer,
    loss=SmoothLoss(0),
    batch_size=args.batch_size,
    sampler=BucketSampler(),
    drop_last=False,
    update_every=32 // args.batch_size,
    num_workers=2,
    n_epochs=args.n_epochs,
    print_every=5,
    dev_data=data_bundle.get_dataset("test"),
    metrics=[AccuracyMetric(), ClassifyFPreRecMetric(f_type="macro")],
    metric_key=None,
    validate_every=-1,
    save_path=None,
    use_tqdm=False,
    device=0,
    callbacks=callbacks,
    check_code_level=0,
    test_sampler=SortedSampler(),
    test_use_tqdm=False,
)

trainer.train(load_best_model=True)


if args.save_embed:
    fitlog.add_other(trainer.start_time, name="start_time")
    os.makedirs(f"{root_fp}/save_models", exist_ok=True)
    folder = f"{root_fp}/save_models/{model_type}-{args.dataset}-FT"
    count = 0
    for fn in os.listdir(f"{root_fp}/save_models"):
        if fn.startswith(folder.split("/")[-1]):
            count += 1
    folder = folder + str(count)
    fitlog.add_other(count, name="count")
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        if model_type  in ('bert', 'roberta'):
            embed.save(folder)
        else:
            embed.save_pretrained(folder)
