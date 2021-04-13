This is modified from the [RGAT](https://github.com/shenwzh3/RGAT-ABSA), which is the source code of the paper [Relational Graph Attention Network for Aspect-based Sentiment Analysis](https://arxiv.org/abs/2004.12362).

We made neccessary changes. We believe all the changes are under the MIT License permission.

## Usage

0. For Glove Embedding

First, download and unzip GloVe vectors(`glove.840B.300d.zip`) from https://nlp.stanford.edu/projects/glove/. Then change the value of parameter `--glove_dir` to the directory of the word vector file.

1. For BERT Embedding

Download the pytorch version pre-trained `bert-base-uncased` model and vocabulary from the link provided by huggingface. Then change the value of parameter `--bert_model_dir` to the directory of the bert model.

2. Train with command

```bash
python run.py --highway --dataset_name the/dataset/path
```

- the `--dataset` arguement should be the path of dataset.

## Notes

0. RGAT model take input files in different format, which should be generated after the running of the Perturbed-Masking code.

1. The `--dataset` should be the data path (/user/project/dataset/Resaurant) rather than the dataset name:

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
