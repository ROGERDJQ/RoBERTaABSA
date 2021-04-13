# ASGCN

This repo is modified from [ASGCN](https://github.com/GeneZC/ASGCN), which is the source code of the
[EMNLP 2019](https://www.emnlp-ijcnlp2019.org/program/accepted/) paper titled "[Aspect-based Sentiment Classification with Aspect-specific Graph Convolutional Networks](https://arxiv.org/abs/1909.03477)".

We made neccessary changes. We opensource all the changes we have made.

## Usage
0. Download pretrained GloVe embeddings from [here](http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip) and extract glove.840B.300d.txt into `glove/`.

1. Install [SpaCy](https://spacy.io/) package and language models with

```bash
pip install spacy
```

and

```bash
python -m spacy download en
```

2. Train with command

```bash
python train.py --dataset the/dataset/path  --save True
```

- the `--dataset` arguement should be the path of dataset.

## Notes

0. ASGCN model take input files in different format, which should be generated after the running of the Perturbed-Masking code.

1. The `--dataset` should be the data path (/user/project/dataset/Resaurant)  rather than the dataset name:

```bash
/user/project/dataset/
(--dataset) |---Restaurant
            |------Train
            |------Test
(--dataset) |---Laptop
            |------Train
            |------Test
(--dataset) |---fr
            |------Train
            |------Test
...
```

## Citation

If you use the code, please cite the paper

```bibtex
@inproceedings{zhang-etal-2019-aspect,
    title = "Aspect-based Sentiment Classification with Aspect-specific Graph Convolutional Networks",
    author = "Zhang, Chen and Li, Qiuchi and Song, Dawei",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov, year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-1464",
    doi = "10.18653/v1/D19-1464",
    pages = "4560--4570",
}
```
