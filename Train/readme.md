The code is mainly used for finetuning pre-train models on ABSA datasets, where the `pipe.py` is the data preprocessing code; the `finetune.py` is used to fine-tune and get the fine-tuned model.

## Usage

1. Install the fastNLp and the fitlog. Our code heavily rely on these two packages.

```bash
pip install fastNLP fitlog
```

2. Run the code with command:

```bash
  python finetune.py --data_dir /user/project/dataset/ --dataset Restaurant
```

- The `--data_dir` and `--dataset` arguments are explained in the following `Notes`.

## Notes

0. The code are based on fastNLP and Fitlog. More can be found in [fastNLP](https://fastnlp.readthedocs.io/zh/latest/) and [fitlog](https://fitlog.readthedocs.io/zh/latest/).
1. The dataset path in our original code is:

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

2. The `Train/Test.json` data files are in the following format：

```json
{
  "sentence": "BEST spicy tuna roll , great asian salad .",
  "token": [
    "BEST",
    "spicy",
    "tuna",
    "roll",
    ",",
    "great",
    "asian",
    "salad",
    "."
  ],
  "pos": [
    "PROPN",
    "ADJ",
    "NOUN",
    "NOUN",
    "PUNCT",
    "ADJ",
    "ADJ",
    "NOUN",
    "PUNCT"
  ],
  "deprel": ["dep", "root", "dep", "dep", "dep", "dep", "dep", "dep", "punct"],
  "head": [2, 0, 4, 2, 4, 5, 8, 6, 7],
  "dependencies": [
    ["dep", 2, 1],
    ["root", 0, 2],
    ["dep", 4, 3],
    ["dep", 2, 4],
    ["dep", 4, 5],
    ["dep", 5, 6],
    ["dep", 8, 7],
    ["dep", 6, 8],
    ["punct", 7, 9]
  ],
  "aspects": [
    { "term": ["asian", "salad"], "polarity": "positive", "from": 6, "to": 8 },
    {
      "term": ["spicy", "tuna", "roll"],
      "polarity": "positive",
      "from": 1,
      "to": 4
    }
  ]
}
```
