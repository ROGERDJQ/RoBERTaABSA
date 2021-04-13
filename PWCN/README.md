# PWCN

This repo is modified from [PWCN](https://github.com/GeneZC/PWCN), which is the source code of the
[SIGIR 2019](https://sigir.org/sigir2019/) paper titled "[Syntax-Aware Aspect-Level Sentiment Classification with Proximity-Weighted Convolution Network](https://arxiv.org/abs/1909.10171)".
We made neccessary changes. We opensource all the changes we have made.

## Usage

0. The `glove/` here contains same files with the `ASGCN` one.

1. Train with command

```bash
python train.py --model_name pwcn_dep --dataset the/dataset/path
```

- the `--dataset` arguement should be the path of dataset.

## Notes

0. PWCN model take input files in different format, which should be generated after the running of the Perturbed-Masking code.

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

## Citation

If you use the code , please cite the paper

```bibtex
@inproceedings{Zhang:2019:SAS:3331184.3331351,
 author = {Zhang, Chen and Li, Qiuchi and Song, Dawei},
 title = {Syntax-Aware Aspect-Level Sentiment Classification with Proximity-Weighted Convolution Network},
 booktitle = {Proceedings of the 42Nd International ACM SIGIR Conference on Research and Development in Information Retrieval},
 series = {SIGIR'19},
 year = {2019},
 isbn = {978-1-4503-6172-9},
 location = {Paris, France},
 pages = {1145--1148},
 numpages = {4},
 url = {http://doi.acm.org/10.1145/3331184.3331351},
 doi = {10.1145/3331184.3331351},
 acmid = {3331351},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {proximity-weighted convolution, sentiment classification, syntax-awareness},
}
```
