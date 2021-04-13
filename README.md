# RoBERTaABSA

This repo contains the code for [NAACL 2021](https://2021.naacl.org/program/accepted/) paper titled [Does syntax matter? A strong baseline for Aspect-based Sentiment Analysis with RoBERTa](https://arxiv.org/abs/2104.04986).

For any questions about the implementation, feel free to create an issue or email me via jiqdai19@fudan.edu.cn.

## Usage

0. Fine-tuning the model on ABSA datasets using the code form `finetune` folder, which will save the fine-tuned models.
1. Generate the induced trees using the code form `Perturbed-Masking` folder, which will output the input datasets for different models.
2. Run the code in `ASGCN`, `PWCN` and `RGAT`.

## Disclaimer

We made neccessary changes based on the original code. We believe all the changes are under the MIT License permission. And we opensource all the changes we have made.
