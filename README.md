# RoBERTaABSA

Implementation for paper **Does syntax matter? A strong baseline for Aspect-based Sentiment Analysis with RoBERTa** [NAACL 2021], a work focusing on Aspect-level Sentiment Classification (ALSC). It conducts a detailed study on the performance gain of dependency tree in ALSC, and provides a strong baseline using RoBERTa.

You can find more details here:

- [`Arxiv`](https://arxiv.org/abs/2104.04986)
- [`Github`](https://github.com/ROGERDJQ/RoBERTaABSA)
- [`Paperwithcode`](https://www.paperswithcode.com/paper/does-syntax-matter-a-strong-baseline-for)
- [`Explainaboard`](http://explainaboard.nlpedia.ai/leaderboard/task-absa/)

For any questions about codes or paper, feel free to create issues or email me via jqdai19@fudan.edu.cn.

For research on the whole ABSA task, please have a look at our ACL 2021 paper [A Unified Generative Framework for Aspect-Based Sentiment Analysis](https://arxiv.org/abs/2106.04300).

## Dependencies

We recommend to create a virtual environment.

```
conda create -n absa
conda activate absa
```

packages:

- python 3.7
- pytorch 1.5.1
- transformers 2.11.0
- fastNLP
  - pip install git+https://github.com/fastnlp/fastNLP@dev
- fitlog
  - pip install git+https://github.com/fastnlp/fitlog

All codes are tested on linux only.

## Data

English Datasets are released in `Dataset` folder for reproduction. If you want to process with your own data, please refer to python scripts in `Dataset` folder.

## Usage

**To get ALSC result:**

To get ALSC result (see [`Paperwithcode`](https://www.paperswithcode.com/paper/does-syntax-matter-a-strong-baseline-for)), simply run the `finetune.py` in `Train` folder. Before the code running, make sure that `--data_dir` and `--dataset` are filled with correct dataset filepath and dataset name.

We also provide detailed arguments and training logs here:

| datasets     | Acc   | F1    | args                                                                           | logs                                                                           |
| ------------ | ----- | ----- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| Restaurant14 | 88.84 | 83.09 | [args](https://github.com/ROGERDJQ/RoBERTaABSA/blob/main/Train/exps/rest_args) | [logs](https://github.com/ROGERDJQ/RoBERTaABSA/blob/main/Train/exps/rest_logs) |
| Laptop14     | 85.1  | 82.08 | [args](https://github.com/ROGERDJQ/RoBERTaABSA/blob/main/Train/exps/lap_args)  | [logs](https://github.com/ROGERDJQ/RoBERTaABSA/blob/main/Train/exps/lap_logs)  |
| Twitter      | 77.46 | 76.87 | [args](https://github.com/ROGERDJQ/RoBERTaABSA/blob/main/Train/exps/twi_args)  | [logs](https://github.com/ROGERDJQ/RoBERTaABSA/blob/main/Train/exps/twi_logs)  |

It is worth noting that the above results are only from one experiment each rather than averaged runs reported in the paper. 

**To reproduce all experiments in paper:**

It includes four steps to reproduce all experiments in our paper:

1. Fine-tuning  models on ABSA datasets using codes from  `Train` folder, which will save the fine-tuned models after fine-tuning.

   ```bash
   python finetune.py --data_dir {/your/dataset_filepath/} --dataset {dataset_name}
   ```

2. Generate the induced trees using the code from the `Perturbed-Masking` folder, which will output the input datasets for different models.

   ```bash
   python generate_matrix.py --model_path bert --data_dir /user/project/dataset/ --dataset Restaurant
   ```

- `model_path` can be either `bert/roberta/xlmroberta/xlmbert`, or the model path where the fine-tuned model is put.

3. Generate corresponding input data for specific model.

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

4. Run the code in `ASGCN`, `PWCN` and `RGAT`.

## Disclaimer

- We made necessary changes based on their original code for `ASGCN`, `PWCN` , `RGAT` and `Perturbed-Masking`. We believe all the changes are under the MIT License permission. And we opensource all the changes we have made.
- At the same time, we try to maintain the original structure of these code. This may lead to errors if running them in their original steps. We recommand you run their code (`ASGCN`, `PWCN` , `RGAT` and `Perturbed-Masking`) following the readme description rather than their original steps.

## Notes

We notice that the learning rate in the paper got mistakes. Please refer to the learning rate in code, which is 2e-5 for RoBERTa.

## Reference

If you find anything interesting about this paper, feel free to cite：

```
@inproceedings{DBLP:conf/naacl/DaiYSLQ21,
  author    = {Junqi Dai and
               Hang Yan and
               Tianxiang Sun and
               Pengfei Liu and
               Xipeng Qiu},
  editor    = {Kristina Toutanova and
               Anna Rumshisky and
               Luke Zettlemoyer and
               Dilek Hakkani{-}T{\"{u}}r and
               Iz Beltagy and
               Steven Bethard and
               Ryan Cotterell and
               Tanmoy Chakraborty and
               Yichao Zhou},
  title     = {Does syntax matter? {A} strong baseline for Aspect-based Sentiment
               Analysis with RoBERTa},
  booktitle = {Proceedings of the 2021 Conference of the North American Chapter of
               the Association for Computational Linguistics: Human Language Technologies,
               {NAACL-HLT} 2021, Online, June 6-11, 2021},
  pages     = {1816--1829},
  publisher = {Association for Computational Linguistics},
  year      = {2021},
  url       = {https://doi.org/10.18653/v1/2021.naacl-main.146},
  doi       = {10.18653/v1/2021.naacl-main.146},
}
```
