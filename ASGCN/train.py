# -*- coding: utf-8 -*-

import os
import math
import argparse
import random
import numpy
import torch
import torch.nn as nn
from bucket_iterator import BucketIterator
from sklearn import metrics
from data_utils import ABSADatesetReader
from models import LSTM, ASCNN, ASGCN
import fitlog

import os

fitlog.debug()
if "p" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["p"]


class Instructor:
    def __init__(self, opt):
        self.opt = opt

        absa_dataset = ABSADatesetReader(
            dataset=opt.dataset, embed_dim=opt.embed_dim, refresh=opt.refresh
        )

        self.train_data_loader = BucketIterator(
            data=absa_dataset.train_data, batch_size=opt.batch_size, shuffle=True
        )
        self.test_data_loader = BucketIterator(
            data=absa_dataset.test_data, batch_size=opt.batch_size, shuffle=False
        )

        self.model = opt.model_class(absa_dataset.embedding_matrix, opt).to(opt.device)
        self._print_args()
        self.global_f1 = 0.0

        if torch.cuda.is_available():
            print(
                "cuda memory allocated:",
                torch.cuda.memory_allocated(device=opt.device.index),
            )

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print(
            "n_trainable_params: {0}, n_nontrainable_params: {1}".format(
                n_trainable_params, n_nontrainable_params
            )
        )
        print("> training arguments:")
        for arg in vars(self.opt):
            print(">>> {0}: {1}".format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for p in self.model.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    self.opt.initializer(p)
                else:
                    stdv = 1.0 / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer):
        max_test_acc = 0
        max_test_f1 = 0
        global_step = 0
        continue_not_increase = 0
        for epoch in range(self.opt.num_epoch):
            print(">" * 100)
            print("epoch: ", epoch)
            n_correct, n_total = 0, 0
            increase_flag = False
            for i_batch, sample_batched in enumerate(self.train_data_loader):
                global_step += 1

                # switch model to training mode, clear gradient accumulators
                self.model.train()
                optimizer.zero_grad()

                inputs = [
                    sample_batched[col].to(self.opt.device)
                    for col in self.opt.inputs_cols
                ]
                targets = sample_batched["polarity"].to(self.opt.device)

                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                if global_step % self.opt.log_step == 0:
                    n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                    n_total += len(outputs)
                    train_acc = n_correct / n_total

                    test_acc, test_f1 = self._evaluate_acc_f1()
                    ################fitlog code####################
                    fitlog.add_metric(test_acc, name="acc", step=global_step)
                    fitlog.add_metric(test_f1, name="f1", step=global_step)
                    ################fitlog code####################
                    if test_acc > max_test_acc:
                        increase_flag = True
                        fitlog.add_best_metric(test_acc, "acc")
                        max_test_acc = test_acc
                    if test_f1 > max_test_f1:
                        increase_flag = True
                        max_test_f1 = test_f1
                        fitlog.add_best_metric(max_test_f1, "f1")
                        if self.opt.save and test_f1 > self.global_f1:
                            self.global_f1 = test_f1
                            torch.save(
                                self.model.state_dict(),
                                "state_dict/"
                                + self.opt.model_name
                                + "_"
                                + self.opt.dataset
                                + ".pkl",
                            )
                            print(">>> best model saved.")
                    print(
                        "loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, test_f1: {:.4f}".format(
                            loss.item(), train_acc, test_acc, test_f1
                        )
                    )
            if increase_flag == False:
                if continue_not_increase >= self.opt.early_stop:
                    print("early stop.")
                    break
                continue_not_increase += 1
            else:
                continue_not_increase = 0
        return max_test_acc, max_test_f1

    def _evaluate_acc_f1(self):
        # switch model to evaluation mode
        self.model.eval()
        n_test_correct, n_test_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(self.test_data_loader):
                t_inputs = [
                    t_sample_batched[col].to(opt.device) for col in self.opt.inputs_cols
                ]
                t_targets = t_sample_batched["polarity"].to(opt.device)
                t_outputs = self.model(t_inputs)

                n_test_correct += (
                    (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                )
                n_test_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        test_acc = n_test_correct / n_test_total
        f1 = metrics.f1_score(
            t_targets_all.cpu(),
            torch.argmax(t_outputs_all, -1).cpu(),
            labels=[0, 1, 2],
            average="macro",
        )
        return test_acc, f1

    def run(self, repeats=1):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()

        max_test_acc_avg = 0
        max_test_f1_avg = 0
        for i in range(repeats):
            print("repeat: ", (i + 1))
            self._reset_params()
            _params = filter(lambda p: p.requires_grad, self.model.parameters())
            optimizer = self.opt.optimizer(
                _params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg
            )
            max_test_acc, max_test_f1 = self._train(criterion, optimizer)
            print(
                "max_test_acc: {0}     max_test_f1: {1}".format(
                    max_test_acc, max_test_f1
                )
            )
            max_test_acc_avg += max_test_acc
            max_test_f1_avg += max_test_f1
            print("#" * 100)
            fitlog.finish()
        print("max_test_acc_avg:", max_test_acc_avg / repeats)
        print("max_test_f1_avg:", max_test_f1_avg / repeats)


if __name__ == "__main__":
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
    )
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument("--l2reg", default=0.00001, type=float)
    parser.add_argument("--num_epoch", default=100, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--embed_dim", default=300, type=int)
    parser.add_argument("--hidden_dim", default=300, type=int)
    parser.add_argument("--dropout", default=0.7, type=float)

    opt = parser.parse_args()  # opt--->all args
    if opt.dataset.endswith("/"):
        opt.dataset = opt.dataset[:-1]
    ################fitlog code####################
    fitlog.set_log_dir("logs")
    fitlog.set_rng_seed()
    fitlog.add_hyper(opt)
    fitlog.add_hyper(value="ASGCN", name="model")
    ################fitlog code####################
    opt.polarities_dim = 3
    opt.initializer = "xavier_uniform_"
    opt.optimizer = "adam"
    opt.model_name = "asgcn"
    opt.log_step = 20
    opt.l2reg = 1e-5
    opt.early_stop = 25

    if "/" in opt.dataset:
        pre_model_name, layer, dataset = opt.dataset.split("/")[-3:]
    else:
        pre_model_name, dataset = "None", opt.dataset
        layer = "0"
    fitlog.add_hyper(value=pre_model_name, name="model_name")
    fitlog.add_hyper(value=dataset, name="dataset")
    fitlog.add_hyper(value=layer, name="pre_layer")

    model_classes = {
        "lstm": LSTM,
        "ascnn": ASCNN,
        "asgcn": ASGCN,
        "astcn": ASGCN,
    }
    input_colses = {
        "lstm": ["text_indices"],
        "ascnn": ["text_indices", "aspect_indices", "left_indices"],
        "asgcn": ["text_indices", "aspect_indices", "left_indices", "dependency_graph"],
        "astcn": ["text_indices", "aspect_indices", "left_indices", "dependency_graph"],
    }
    initializers = {
        "xavier_uniform_": torch.nn.init.xavier_uniform_,
        "xavier_normal_": torch.nn.init.xavier_normal_,
        "orthogonal_": torch.nn.init.orthogonal_,
    }
    optimizers = {
        "adadelta": torch.optim.Adadelta,  # default lr=1.0
        "adagrad": torch.optim.Adagrad,  # default lr=0.01
        "adam": torch.optim.Adam,  # default lr=0.001
        "adamax": torch.optim.Adamax,  # default lr=0.002
        "asgd": torch.optim.ASGD,  # default lr=0.01
        "rmsprop": torch.optim.RMSprop,  # default lr=0.01
        "sgd": torch.optim.SGD,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt.save = False
    opt.refresh = False

    ins = Instructor(opt)
    ins.run()
