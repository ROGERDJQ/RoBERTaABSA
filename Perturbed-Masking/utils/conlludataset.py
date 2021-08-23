import io
import os

from utils import UToken

ROOT_TOKEN = '<root>'
ROOT_TAG = 'ROOT'
ROOT_LABEL = '-root-'


def empty_conllu_example_dict():
    ex = {
        'id':      [],
        'form':    [],
        'lemma':   [],
        'pos':     [],
        'upos':    [],
        'feats':   [],
        'head':    [],
        'deprel':  [],
        'deps':    [],
        'misc':    [],
    }
    return ex


def root():
    ex = {
        'id': [0],
        'form': [ROOT_TOKEN],
        'lemma': [ROOT_TOKEN],
        'pos': [ROOT_TAG],
        'upos': [ROOT_TAG],
        'feats': ['_'],
        'head': [0],
        'deprel': [ROOT_LABEL],
        'deps': ['_'],
        'misc': ['_'],
    }
    return ex


def conllu_reader(f):
    """
    file中的格式类似与

    1	“	“	PUNCT	``	_	20	punct	20:punct	SpaceAfter=No
    2	While	while	SCONJ	IN	_	9	mark	9:mark	_
    3	much	much	ADJ	JJ	Degree=Pos	9	nsubj	9:nsubj	_
    4	of	of	ADP	IN	_	7	case	7:case	_

    :param f:
    :return:
    """
    ex = root()

    for line in f:
        line = line.strip()

        if not line:
            yield ex
            ex = root()
            continue

        # comments
        if line[0] == "#":
            continue

        parts = line.split()
        assert len(parts) == 10, "invalid conllx line: %s" % line

        _id, _form, _lemma, _upos, _xpos, _feats, _head, _deprel, _deps, _misc = parts

        ex['id'].append(_id)
        ex['form'].append(_form)  # 就是原始的词
        ex['lemma'].append(_lemma) # lemmatize之后的词
        ex['upos'].append(_upos)
        ex['pos'].append(_xpos)
        ex['feats'].append(_feats)

        # TODO: kan dit? (0 is root)
        if _head == "_":
            _head = 0

        ex['head'].append(_head)  # 词语的id
        ex['deprel'].append(_deprel)  # nsbj
        ex['deps'].append(_deps)  # 例如9:nsbj
        ex['misc'].append(_misc)

    # possible last sentence without newline after
    if len(ex['form']) > 0:
        yield ex


class ConllUDataset:
    """Defines a CONLL-U Dataset. """

    def __init__(self, path):
        print('dataset used:', os.path.expanduser(path))
        """Create a ConllUDataset given a path and field list.
        Arguments:
            path (str): Path to the data file.
            fields (dict[str: tuple(str, Field)]):
                The keys should be a subset of the columns, and the
                values should be tuples of (name, field).
                Keys not present in the input dictionary are ignored.
        """

        with io.open(os.path.expanduser(path), encoding="utf8") as f:
            self.examples = [d for d in conllu_reader(f)]

        # for d in self.examples:
        #     token = []
        #     for parts in zip(*d.values()):
        #         token.append(UToken(*parts))
            # self.tokens.append(token)

        # this line is a pythonic version of the above fuction
        # d is a whole sentence
        # d.values gets all values of the dict
        # 单星号（*）：*agrs 将所以参数以元组(tuple)的形式导入：
        # parts represents one line in conllu file
        # 只需要解析出tokens里面的
        self.tokens = [
            [UToken(*parts) for parts in zip(*d.values())]
            for d in self.examples]

from fastNLP.io import Pipe, Loader, DataBundle
from fastNLP import DataSet, Instance
import json


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
        fns = [f'{folder_name}_Test_biaffine_depparsed.json',
               f'{folder_name}_Train_biaffine_depparsed.json']
        if not os.path.exists(os.path.join(paths, fns[0])):
            fns = [f'Test.json',
                   f'Train.json']

        for split, name in zip(['test', 'train'], fns):
            fp = os.path.join(paths, name)
            with open(fp, 'r', encoding='utf-8') as f:
                data = json.load(f)
            ds = DataSet()
            for ins in data:
                tokens = ins['token']
                pos = ins['pos']
                dep = ins['dependencies']
                aspects = ins['aspects']
                ins = Instance(tokens=tokens, pos=pos, dep=dep, aspects=aspects)
                ds.append(ins)
            data_bundle.set_dataset(ds, name=split)
        return data_bundle


# c = ConllUDataset('./data/EWT/en_ewt-ud-test.conllu')
# print('done')