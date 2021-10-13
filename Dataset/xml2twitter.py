
import json
from typing import Dict
from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from nltk.tokenize import (
    TreebankWordTokenizer,
    word_tokenize,
    ToktokTokenizer,
    PunktSentenceTokenizer,
)
import allennlp_models.structured_prediction
from copy import deepcopy
from lxml import etree
from tqdm import tqdm


def txt2format(file_path):
    # [sent,asp,pol,[f,t]]
    with open(file_path, "r", encoding="utf8") as f:
        raw = f.readlines()
        target_dict = {"0": "neutral", "1": "positive", "-1": "negative"}
        ff = [
            [
                raw[i * 3].strip(),
                raw[3 * i + 1].strip(),
                raw[3 * i + 2].strip(),
                [
                    raw[i * 3].strip().find("$T$"),
                    raw[i * 3].strip().find("$T$") + len(raw[3 * i + 1].strip()),
                ],
            ]
            for i in range(len(raw) // 3)
        ]
        formatfile = [[i[0].replace("$T$", i[1]), i[1], i[2], i[3]] for i in ff]
        formatfile = [[i[0], i[1], target_dict[i[2]], i[3]] for i in formatfile]
    return formatfile


def format2json(predictor, file_path):
    ff = txt2format(file_path)
    tk = SpacyTokenizer()
    output = []
    for sent in tqdm(ff):
        example = {}
        example["sentence"] = sent[0]
        doc = predictor.predict(sentence=sent[0])
        token = doc["words"]
        pos = doc["pos"]
        # sentence['energy'] = doc['energy']
        predicted_dependencies = doc["predicted_dependencies"]
        predicted_heads = doc["predicted_heads"]
        deprel = doc["predicted_dependencies"]
        head = doc["predicted_heads"]
        dependencies = []
        for idx, item in enumerate(predicted_dependencies):
            dep_tag = item
            frm = predicted_heads[idx]  
            to = idx + 1
            dependencies.append([dep_tag, frm, to])  #
        example["token"] = list(token)
        example["pos"] = pos
        example["deprel"] = deprel
        example["head"] = head
        example["dependencies"] = dependencies
        example["aspects"] = []
        asp = dict()
        asp["term"] = [str(i) for i in tk.tokenize(sent[1])]
        asp["polarity"] = sent[2]
        asp["from"] = len(tk.tokenize(sent[0][: sent[3][0]]))
        asp["to"] = len(tk.tokenize(sent[0][: sent[3][1]]))
        example["aspects"].append(asp)
        output.append(example)
    extended_filename = file_path.replace(".raw", "_biaffine_depparsed.json")
    with open(extended_filename, "w") as f:
        json.dump(output, f, indent=2)
    print("done", len(output))


def get_all(path, predictor):
    format2json(predictor, path + "/test.raw")
    format2json(predictor, path + "/train.raw")


predictor = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz"
)

get_all("Dataset/Tweets", predictor)
