
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

from lxml import etree
from tqdm import tqdm


def xml2txt(file_path, predictor):
    """
    Read the original xml file of semeval data and extract the text that have aspect terms.
    Store them in txt file.
    file_path: origin_file_path
    """
    tk = SpacyTokenizer()

    with open(file_path, "rb") as f:
        raw = f.read()
        root = etree.fromstring(raw)
        final_out = []
        for sentence in root:
            example = {}
            sent = sentence.find("text").text
            terms = sentence.find("aspectTerms")
            if terms is None:  
                continue
            if terms is not None:
                sent_list = list(sent)
                fidx_list = []
                tidx_list = []
                for t in terms:
                    if t.attrib["polarity"] == "conflict":
                        continue
                    fidx = int(t.attrib["from"])
                    tidx = int(t.attrib["to"])
                    assert tidx <= len(sent_list)
                    fidx_list.append(fidx)  # like [22, 36]
                    tidx_list.append(tidx)  # like [31, 57]
                if len(fidx_list) == 0:  
                    continue
                insert_idx = sorted(fidx_list + tidx_list, reverse=True)
                for idx in insert_idx:  
                    sent_list.insert(idx, " ")
                fidx_list = sorted(fidx_list)
                tidx_list = sorted(tidx_list)
                sent = "".join(sent_list)
                example["sentence"] = sent
                allen = predictor.predict(sentence=sent)
                token, pos, deprel, head, dependencies = dependencies2format(allen)
                example["token"] = list(token)
                example["pos"] = pos
                example["deprel"] = deprel
                example["head"] = head
                example["dependencies"] = dependencies
                example["aspects"] = []
                for t in terms:
                    if t.attrib["polarity"] == "conflict":
                        continue
                    asp = dict()
                    asp["term"] = [str(i) for i in tk.tokenize(t.attrib["term"])] 
                    asp["polarity"] = t.attrib["polarity"]
                    assert fidx_list.index(int(t.attrib["from"])) == tidx_list.index(
                        int(t.attrib["to"])
                    )  
                    left_index = (
                        int(t.attrib["from"])
                        + 1
                        + 2 * (fidx_list.index(int(t.attrib["from"])))
                    )
                    right_index = (
                        int(t.attrib["to"])
                        + 1
                        + 2 * (tidx_list.index(int(t.attrib["to"])))
                    )
                    left_word_offset = len(tk.tokenize(sent[:left_index]))
                    to_word_offset = len(tk.tokenize(sent[:right_index]))
                    asp["from"] = left_word_offset
                    asp["to"] = to_word_offset
                    example["aspects"].append(asp)
                final_out.append(example)
        extended_filename = file_path.replace(".xml", "_biaffine_depparsed.json")
        with open(extended_filename, "w") as f:
            json.dump(final_out, f, indent=2)
    print("done", len(final_out))


def dependencies2format(doc):  # doc.sentences[i]
    """
    Format annotation: sentence of keys
                                - tokens
                                - tags
                                - predicted_dependencies
                                - predicted_heads
                                - dependencies
    RETURN token,pos,deprel,head,dependencies
    """
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
        dependencies.append(
            [dep_tag, frm, to]
        )  

    return token, pos, deprel, head, dependencies


def get_all_file(path):
    base_dir = path
    xml2txt(path + "_Test.xml", predictor)
    xml2txt(path + "_Train.xml", predictor)


predictor = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz"
)


get_all_file(
    "Dataset/Restaurants"
)
