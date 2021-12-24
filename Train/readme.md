This folder  mainly contains codes for fine-tuning  models on ABSA datasets.
- `pipe.py` pre-processes data. 
- `finetune.py` fine-tunes models, which  also  save the fine-tuned models after fine-tuning.

## Usage

1. Install fastNLP and fitlog. Our codes  rely on this two packages.



2. Run the codes with command:

```bash
  python finetune.py 
  --data_dir {/your/dataset_filepath/} 
  --dataset {dataset_name}
```


## Notes

0. The codes are based on fastNLP and Fitlog. More can be found in [fastNLP](https://fastnlp.readthedocs.io/zh/latest/) and [fitlog](https://fitlog.readthedocs.io/zh/latest/).
1. The codes will use data files named as  `Train/Test.json`  under the  `{data_dir}/{dataset}` folder, e.g. `{RoBERTaABSA/Dataset}/{Restaurant}`, so please make sure this two files are available.


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
