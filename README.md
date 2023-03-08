# RoBERTaABSA

Implementation for paper **Does syntax matter? A strong baseline for Aspect-based Sentiment Analysis with RoBERTa** [NAACL 2021], a work focusing on Aspect-level Sentiment Classification (ALSC). It conducts a detailed study on the performance gain of dependency tree in ALSC, and provides a strong baseline using RoBERTa.

You can find more information here:

- [`Paper`](https://arxiv.org/abs/2104.04986)
- [`Code`](https://github.com/ROGERDJQ/RoBERTaABSA)
- [`Paperwithcode`](https://www.paperswithcode.com/paper/does-syntax-matter-a-strong-baseline-for)

For any questions about code or paper, feel free to create issues or email me via jqdai19@fudan.edu.cn.

If you are interested in on the whole ABSA task, please have a look at our ACL 2021 paper [A Unified Generative Framework for Aspect-Based Sentiment Analysis](https://arxiv.org/abs/2106.04300).

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
- fastNLP < 1.0
  - pip install git+https://github.com/fastnlp/fastNLP@dev
- fitlog
  - pip install git+https://github.com/fastnlp/fitlog

All codes are tested on linux only.

## Data

English Datasets are released in `Dataset` folder for reproduction. If you want to process with your own data, please refer to scripts in `Dataset` folder.

## Usage

**To get ALSC result:**

To get ALSC results (see [`Paperwithcode`](https://www.paperswithcode.com/paper/does-syntax-matter-a-strong-baseline-for)),  run  `finetune.py` in `Train` folder. Before code running, remember to check `Notes` below, and make sure `--data_dir` and `--dataset` are filled with correct dataset filepath and dataset name.  

We also provide detailed arguments and logs here:

| datasets     | args                                                                           | logs                                                                           |
| ------------  | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| Restaurant14  | [args](https://github.com/ROGERDJQ/RoBERTaABSA/blob/main/Train/exps/rest_args) | [logs](https://github.com/ROGERDJQ/RoBERTaABSA/blob/main/Train/exps/rest_logs) |
| Laptop14     | [args](https://github.com/ROGERDJQ/RoBERTaABSA/blob/main/Train/exps/lap_args)  | [logs](https://github.com/ROGERDJQ/RoBERTaABSA/blob/main/Train/exps/lap_logs)  |
| Twitter      | [args](https://github.com/ROGERDJQ/RoBERTaABSA/blob/main/Train/exps/twi_args)  | [logs](https://github.com/ROGERDJQ/RoBERTaABSA/blob/main/Train/exps/twi_logs)  |

It is worth noting that above results are only tested by one run rather than averaged runs reported in the paper. Also remember to check  `Notes` below.

**To reproduce the whole experiment in paper:**

It includes four steps to reproduce the whole experiment in our paper:

1. Fine-tuning models on ALSC datasets using codes in  `Train` folder, which will save fine-tuned models after fine-tuning.

   ```bash
   python finetune.py --data_dir {/your/dataset_filepath/} --dataset {dataset_name}
   ```

2. Generate induced trees using code in `Perturbed-Masking` folder, which will output datasets serving as  input  for different models.

   ```bash
   python generate_matrix.py --model_path bert --data_dir /user/project/dataset/ --dataset Restaurant
   ```

- `model_path` can be either `bert/roberta/xlmroberta/xlmbert`, or the model path where the above fine-tuned model is put.

3. Generate data with different input format corrsponding to specific model.

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

4. Run code in `ASGCN`, `PWCN` and `RGAT`.

## Disclaimer

- We made necessary changes to the original code of `ASGCN`, `PWCN` , `RGAT` and `Perturbed-Masking`. All the changes we have made are opensourced. We believe all the changes are under MIT License permission. 
- Errors may be raised if run above codes following their original steps. We recommand to run them (`ASGCN`, `PWCN` , `RGAT` and `Perturbed-Masking`) following the readme description in corresponding folders.

## Notes

- The learning rate in the paper was written incorrectly and should be corrected to 2e-5 for RoBERTa.
- Remember to split validation set on your own data.  The "dev" argument should be filled with corresponding validation filepath in the trainer of `finetune.py`. We did not provide a validation set partition here, which was an issue that we previously overlooked. Yet in the implementation of our experiment, we use validation set  to evaluate performance of different induced trees.

## Reference

If you find this work useful, feel free to cite：

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
