# RoBERTaABSA

This repo contains code for [NAACL 2021](https://2021.naacl.org/program/accepted/) paper titled [Does syntax matter? A strong baseline for Aspect-based Sentiment Analysis with RoBERTa](https://arxiv.org/abs/2104.04986).

The summarized information is here:
- [Paper](https://arxiv.org/abs/2104.04986)
- [Code](https://github.com/ROGERDJQ/RoBERTaABSA)
- [Paperwithcode](https://www.paperswithcode.com/paper/does-syntax-matter-a-strong-baseline-for)
- [Explainaboard](http://explainaboard.nlpedia.ai/leaderboard/task-absa/)

For any questions about the implementation, feel free to create an issue or email me via jqdai19@fudan.edu.cn.

## Update 05.21.2021
### The SOTA on RoBERTa
I have received some questions about the reproduction issues. After the re-check of the code and some discussions, we conjecture that the problem may be caused by the different usage of RoBERTa Tokenizer and some pre-processing. We release the datasets in the `Dataset` folder.

To get our RoBERTa results, simply run the `finetune.py` in `Train` folder.  Before the code running, make sure the `--data_dir` and `--dataset` arguments are filled in correct file path.


## The Usage
0. This section is for the reproduction of experiments in the original paper; for the reproduction of RoBERTa SOTA results in [Paperwithcode](https://www.paperswithcode.com/paper/does-syntax-matter-a-strong-baseline-for), please refer to the next section.
1. Fine-tuning the model on ABSA datasets using the code from the `Train` folder, which will save the fine-tuned models.
2. Generate the induced trees using the code from the `Perturbed-Masking` folder, which will output the input datasets for different models.
3. Run the code in `ASGCN`, `PWCN` and `RGAT`.



## Disclaimer
We made necessary changes based on the original code. We believe all the changes are under the MIT License permission. And we opensource all the changes we have made.
