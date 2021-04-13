from fastNLP.io import Pipe, Loader, DataBundle
import os
from fastNLP import DataSet, Instance
import json
from fastNLP import Vocabulary
from fastNLP.modules.tokenizer import RobertaTokenizer, BertTokenizer
import warnings

warnings.filterwarnings("ignore")
from transformers import XLNetTokenizer, XLMRobertaTokenizer


class ResPipe(Pipe):
    def __init__(self, model_name="en", mask="<mask>"):
        super().__init__()
        model_type = "roberta"
        if model_name.split("-")[0] in ("bert", "roberta", "xlnet", "xlmroberta"):
            model_type, model_name = (
                model_name[: model_name.index("-")],
                model_name[model_name.index("-") + 1 :],
            )
        if model_type == "roberta":
            self.tokenizer = RobertaTokenizer.from_pretrained(
                model_dir_or_name=model_name
            )
        elif model_type == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(model_dir_or_name=model_name)
        elif model_type == "xlnet":
            self.tokenizer = XLNetTokenizer.from_pretrained(
                pretrained_model_name_or_path=model_name
            )
        elif model_type == "xlmroberta":
            self.tokenizer = XLMRobertaTokenizer.from_pretrained(
                pretrained_model_name_or_path=model_name
            )

        self.mask = mask

    def process(self, data_bundle: DataBundle) -> DataBundle:
        new_bundle = DataBundle()
        aspect_dict = {}
        mask_id = self.tokenizer.convert_tokens_to_ids([self.mask])[0]
        if isinstance(self.tokenizer, BertTokenizer):
            cls = "[CLS]"
            sep = "[SEP]"
        else:
            cls = self.tokenizer.cls_token
            sep = self.tokenizer.sep_token
        for name, ds in data_bundle.iter_datasets():
            new_ds = DataSet()
            for ins in ds:
                tokens = ins["tokens"]
                if not isinstance(self.tokenizer, XLNetTokenizer):
                    tokens.insert(0, cls)
                    tokens.append(sep)
                    shift = 1
                else:
                    tokens.append(sep)
                    tokens.append(cls)
                    shift = 0

                starts = []
                ends = []
                for aspect in ins["aspects"]:
                    starts.append(aspect["from"] + shift)
                    ends.append(aspect["to"] + shift)
                for aspect in ins["aspects"]:
                    target = aspect["polarity"]
                    start = aspect["from"] + shift
                    end = aspect["to"] + shift
                    aspect_mask = [0] * len(tokens)
                    for i in range(start, end):
                        aspect_mask[i] = 1
                    pieces = []
                    piece_masks = []
                    raw_words = tokens[shift:-1]
                    raw_words.insert(start - 1, "[[")
                    raw_words.insert(end, "]]")
                    for mask, token in zip(aspect_mask, tokens):
                        bpes = self.tokenizer.convert_tokens_to_ids(
                            self.tokenizer.tokenize(token)
                        )
                        pieces.extend(bpes)
                        piece_masks.extend([mask] * (len(bpes)))
                    new_ins = Instance(
                        tokens=pieces,
                        target=target,
                        aspect_mask=piece_masks,
                        raw_words=" ".join(raw_words),
                    )
                    new_ds.append(new_ins)
            new_bundle.set_dataset(new_ds, name)

        target_vocab = Vocabulary(padding=None, unknown=None)
        target_vocab.add_word_lst(["neutral", "positive", "negative", "smooth"])
        target_vocab.index_dataset(*new_bundle.datasets.values(), field_name="target")

        new_bundle.set_target("target")
        new_bundle.set_input("tokens", "aspect_mask", "raw_words")
        new_bundle.apply_field(
            lambda x: len(x), field_name="tokens", new_field_name="seq_len"
        )

        # new_bundle.set_vocab(vocab, 'tokens')
        if hasattr(self.tokenizer, "pad_token_id"):
            new_bundle.set_pad_val("tokens", self.tokenizer.pad_token_id)
        else:
            new_bundle.set_pad_val("tokens", self.tokenizer.pad_index)
        new_bundle.set_vocab(target_vocab, "target")

        return new_bundle

    def process_from_file(self, paths) -> DataBundle:
        data_bundle = ResLoader().load(paths)
        return self.process(data_bundle)


class ResLoader(Loader):
    def __init__(self):
        super().__init__()

    def load(self, paths):
        """
        输出的DataSet包含以下的field
        tokens                  pos                   dep                                    aspects
        ["The", "bread", ...]   ["DET", "NOUN",...]   [["dep", 2, 1], ["nsubj", 4, 2], ...]  [{"term": ["bread"], "polarity": "positive", "from": 1, "to": 2}]
        其中dep中["dep", 2, 1]指当前这个word的head是2（0是root，这里2就是bread），"dep"是依赖关系为dep

        :param paths:
        :return:
        """
        data_bundle = DataBundle()
        folder_name = os.path.basename(paths)
        fns = [
            f"{folder_name}_Test.json",
            f"{folder_name}_Train.json",
        ]
        if not os.path.exists(os.path.join(paths, fns[0])):
            fns = [f"Test.json", f"Train.json"]

        for split, name in zip(["test", "train"], fns):
            fp = os.path.join(paths, name)
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            ds = DataSet()
            for ins in data:
                tokens = ins["token"]
                pos = ins["pos"]
                dep = ins["dependencies"]
                aspects = ins["aspects"]
                ins = Instance(tokens=tokens, pos=pos, dep=dep, aspects=aspects)
                ds.append(ins)
            data_bundle.set_dataset(ds, name=split)
        return data_bundle
