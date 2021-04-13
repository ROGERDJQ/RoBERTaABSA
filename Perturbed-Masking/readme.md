# Perturbed-Masking

This is modified from [Perturbed-Masking](https://github.com/LividWo/Perturbed-Masking), which is the source code of the ACL'2020 Paper

[**Perturbed Masking: Parameter-free Probing for Analyzing and Interpreting BERT**](https://arxiv.org/abs/2004.14786)

We made neccessary changes. We believe all the changes are under the MIT License permission.

## Usage

From the view of our paper, the code is used to generate the induced trees. The general scenario is 1. run `generate_matrix.py` to get the impact matrix; 2. generate input data for different model specifically.

1. Run `generate_matrix.py`.

```bash
python generate_matrix.py --model_path Bert --data_dir /user/project/dataset/ --dataset Restaurant
```

- `model_path` can be either `Bert/RoBERTa/xlmRoberta/xlmbert` or the model path where the fine-tuned model is put.
- `--data_dir` and `--dataset` arguments are explained in the following `Notes`.

2. Generate input data for different model.

- ASGCN Input Data:

```bash
python generate_asgcn.py --layers 11
```

- PWCN Input Data:

```bash
python generate_pwcn.py --layers 11
```

- RGAT Input Data:

```bash
python generate_rgat.py --layers 11
```

## Notes

0. The dataset path in our original code is:

```bash
(--data_dir)/user/project/dataset/
(--dataset) |---Restaurant
            |------Train.json
            |------Test.json
(--dataset) |---Laptop
            |------Train.json
            |------Test.json
(--dataset) |---fr
            |------Train.json
            |------Test.json
...
```

For datasets in `generate_matrix.py`, the data format is like following：

```json
    {
   "sentence": "BEST spicy tuna roll , great asian salad .",
   "token": [ "BEST", "spicy", "tuna", "roll", ",", "great", "asian", "salad", "." ],
   "pos": [ "PROPN", "ADJ", "NOUN", "NOUN", "PUNCT", "ADJ", "ADJ", "NOUN", "PUNCT" ],
   "deprel": [ "dep", "root", "dep", "dep", "dep", "dep", "dep", "dep", "punct" ],
   "head": [ 2, 0, 4, 2, 4, 5, 8, 6, 7 ],
   "dependencies": [
   [ "dep", 2, 1 ],
   [ "root", 0, 2 ],
   [ "dep", 4, 3 ],
   [ "dep", 2, 4 ],
   [ "dep", 4, 5 ],
   [ "dep", 5, 6 ],
   [ "dep", 8, 7 ],
   [ "dep", 6, 8 ],
   [ "punct", 7, 9 ]
   ],
   "aspects": [
   { "term": [ "asian", "salad" ], "polarity": "positive", "from": 6, "to": 8 },
   { "term": [ "spicy", "tuna", "roll" ], "polarity": "positive", "from": 1, "to": 4 }
   ]
   },
```
