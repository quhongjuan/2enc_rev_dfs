# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""
from __future__ import absolute_import

from eval_score.bleu.google_bleu import corpus_bleu
from eval_score.rouge.rouge import Rouge
from eval_score.meteor.meteor import Meteor

#import ssl
#ssl._create_default_https_context = ssl._create_unverified_context


import os
import sys
import bleu
import pickle
import torch
import json
import random
import logging
import argparse
import dill
import numpy as np
from io import open
from itertools import cycle
import torch.nn as nn
from model_codebert_contrastive2 import Seq2Seq
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
logger = logging.getLogger(__name__)

fh = logging.FileHandler(filename='log/log_cbert_contra2.txt', mode='a')

logger.addHandler(fh)


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 dfs,
                 rev,
                 target,
                 source_real_length,
                 pdg_real_length
                 ):
        self.idx = idx
        self.source = source
        self.dfs = dfs
        self.rev = rev
        self.target = target
        self.source_real_length = source_real_length
        self.pdg_real_length = pdg_real_length


def read_examples(filename):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        lines = f.readlines()
        for idx, line in tqdm(enumerate(lines), total=len(lines), desc='read examples'):
            '''if idx > 100:
                print('**********only read 100 examples\n')
                break'''
            line = line.strip() #delete line'space or \n on the start and end
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            # code=' '.join(js['code_tokens']).replace('\n',' ')
            code_token_list = js['code'].strip().split()[:150]
            code_length = len(code_token_list)
            code = ' '.join(code_token_list)

            pdg_token_list = js['dfs'].strip().split()
            pdg_length = len(pdg_token_list)
            dfs = ' '.join(pdg_token_list)

            code += ' <s> ' +dfs

            rev = ' '.join(js['rev'].strip().split())
            # nl=' '.join(js['docstring_tokens']).replace('\n','')
            nl = ' '.join(js['nl'].strip().split())
            examples.append(
                Example(
                    idx=idx,
                    source=code,
                    dfs=dfs,
                    rev=rev,
                    target=nl,
                    source_real_length=code_length,
                    pdg_real_length=pdg_length

                )
            )
    return examples


def create_vocab(filenames, special_token=None):
    if special_token is None:
        special_token = {'<bos>': 0, '<eos>': 2, '<pad>': 1, '<UNK>': 3, '<sep>': 4}
    print('special_token:{}'.format(special_token))
    wordcount = {}
    wordcount_nl = {}
    for filename in filenames:
        with open(filename, 'r') as f:
            lines = f.readlines()
            for i, line in tqdm(enumerate(lines), total=len(lines), desc='wordcount'):
                one_data_dict = json.loads(line)
                for ii, v in enumerate(one_data_dict['code'].split(' ')):
                    if v in wordcount:
                        wordcount[v] += 1
                    else:
                        wordcount.update({v: 1})
                for ii, v in enumerate(one_data_dict['dfs'].split(' ')):
                    if v in wordcount:
                        wordcount[v] += 1
                    else:
                        wordcount.update({v: 1})
                for ii, v in enumerate(one_data_dict['nl'].split(' ')):
                    if v in wordcount:
                        wordcount[v] += 1
                    else:
                        wordcount.update({v: 1})

    sorted_words = sorted(wordcount.items(), key=lambda x: x[1], reverse=True)
    sorted_words_nl = sorted(wordcount_nl.items(), key=lambda x: x[1], reverse=True)

    src_words_to_ids = {}
    src_words_to_ids.update(special_token)
    for i, v in enumerate(sorted_words):
        if i < 50000 - len(special_token):
            src_words_to_ids.update({v[0]: i + len(special_token)})

    tgt_words_to_ids = {}
    tgt_words_to_ids.update(special_token)
    for i, v in enumerate(sorted_words_nl):
        if i < 50000 - len(special_token):
            tgt_words_to_ids.update({v[0]: i + len(special_token)})

    src_ids_to_words = {}
    tgt_ids_to_words = {}
    for i, v in enumerate(src_words_to_ids.items()):
        src_ids_to_words.update({v[1]: v[0]})
    for i, v in enumerate(tgt_words_to_ids.items()):
        tgt_ids_to_words.update({v[1]: v[0]})

    fw = open('./vocab.txt', 'w', encoding='utf-8')
    fw.write(json.dumps(src_words_to_ids) + '\n' + json.dumps(tgt_words_to_ids))

    fw.close()
    tgt_words_to_ids = src_words_to_ids
    tgt_ids_to_words = src_ids_to_words
    return src_words_to_ids, tgt_words_to_ids, src_ids_to_words, tgt_ids_to_words


class Tokenizer(object):

    def __init__(self, dev_file=None, train_file=None, special_token=None):
        self.bos_token = '<bos>'
        self.eos_token = '<eos>'
        self.padding_token = '<pad>'
        self.unkonwn_token = '<UNK>'
        self.split_token = '<bos>'
        self.cls_token = self.bos_token
        self.sep_token = self.eos_token
        self.cls_token_id = 0
        self.sep_token_id = 2
        self.pad_token_id = 1
        self.unkonwn_token_id = 3
        self.split_token_id = 4
        if special_token:
            self.special_token = special_token
        else:
            self.special_token = {self.bos_token: self.cls_token_id, self.eos_token: self.sep_token_id,
                                  self.padding_token: self.pad_token_id, self.unkonwn_token: self.unkonwn_token_id,
                                  self.split_token: self.split_token_id}

        self.src_vocab_w2i, self.tgt_vocab_w2i, self.src_vocab_i2w, self.tgt_vocab_i2w = create_vocab(
            [dev_file, train_file], special_token=self.special_token)
        self.src_vocab_size = len(self.src_vocab_w2i)
        self.tgt_vocab_size = len(self.tgt_vocab_w2i)

    def convert_tokens_to_ids(self, tokens, vocab_kind=None):
        # tokens: ['<bos>', 'public', 'int', 'hash', 'Code', '(', ')', '{', 'return', 'value', '.', 'hash', 'Code', '(', ')', ';', '}', '<eos>']
        # return: [1, 15110, 6979, 29524, 8302, 36, 4839, 25522, 671, 923, 479, 29524, 8302, 36, 4839, 25606, 35524, 2]
        if vocab_kind is None:
            return [self.src_vocab_w2i[w] if w in self.src_vocab_w2i else self.src_vocab_w2i[self.unkonwn_token] for w
                    in tokens]
        if vocab_kind == 'src':
            return [self.src_vocab_w2i[w] if w in self.src_vocab_w2i else self.src_vocab_w2i[self.unkonwn_token] for w
                    in tokens]
        elif vocab_kind == 'tgt':
            return [self.tgt_vocab_w2i[w] if w in self.tgt_vocab_w2i else self.tgt_vocab_w2i[self.unkonwn_token] for w
                    in tokens]

    def decode(self, ids, clean_up_tokenization_spaces=False, vocab_kind='tgt'):
        # ids: list:[5,5,5,5,5....,5] 200
        # return 'the the the ....the ' the*200
        if vocab_kind == 'src':
            return ' '.join([self.src_vocab_i2w[ind] if ind in self.src_vocab_i2w else self.unkonwn_token for ind in ids])
        else:
            return ' '.join([self.tgt_vocab_i2w[ind] if ind in self.tgt_vocab_i2w else self.unkonwn_token for ind in ids])

    def tokenize(self, src):
        # src: 'hash code return value . hash code ( )'
        # return ['public', 'int', 'hash', 'Code', '(', ')', '{', 'return', 'value', '.', 'hash', 'Code', '(', ')', ';', '}']

        return src.split(' ')


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 dfs_ids,
                 rev_ids,
                 target_ids,
                 source_mask,
                 dfs_mask,
                 rev_mask,
                 target_mask,

                 source_real_length,
                 pdg_real_length

                 ):
        self.example_id = example_id

        self.source_ids = source_ids
        self.dfs_ids = dfs_ids
        self.rev_ids = rev_ids
        self.target_ids = target_ids

        self.source_mask = source_mask
        self.dfs_mask = dfs_mask
        self.rev_mask = rev_mask
        self.target_mask = target_mask
        self.source_real_length = source_real_length
        self.pdg_real_length = pdg_real_length


def convert_examples_to_features(examples, tokenizer, args, stage=None):
    features = []
    for example_index, example in tqdm(enumerate(examples), total=len(examples), desc='convert_examples_to_features'):

        # source
        # source file
        source_tokens = tokenizer.tokenize(example.source)[:args.max_source_length - 2] # ['public', 'int', 'hash', 'Code', '(', ')', '{', 'return', 'value', '.', 'hash', 'Code', '(', ')', ';', '}']
        source_tokens = [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]
        #source_ids = tokenizer.convert_tokens_to_ids(source_tokens, vocab_kind='src') # tran
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        source_mask = [1] * (len(source_tokens))
        padding_length = args.max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        source_mask += [0] * padding_length

        # dfs 
        dfs_tokens = tokenizer.tokenize(example.dfs)[:200 - 2]
        dfs_tokens = [tokenizer.cls_token] + dfs_tokens + [tokenizer.sep_token]
        #dfs_ids = tokenizer.convert_tokens_to_ids(dfs_tokens, vocab_kind='src') # tran
        dfs_ids = tokenizer.convert_tokens_to_ids(dfs_tokens)
        dfs_mask = [1] * (len(dfs_tokens))
        padding_length = 200 - len(dfs_ids)
        dfs_ids += [tokenizer.pad_token_id] * padding_length
        dfs_mask += [0] * padding_length

        ## review comments
        rev_tokens = tokenizer.tokenize(example.rev)[:200 - 2]
        rev_tokens = [tokenizer.cls_token] + rev_tokens + [tokenizer.sep_token]
        #rev_ids = tokenizer.convert_tokens_to_ids(rev_tokens, vocab_kind='src')  #tran
        rev_ids = tokenizer.convert_tokens_to_ids(rev_tokens)
        rev_mask = [1] * (len(rev_tokens))
        padding_length = 200 - len(rev_ids)
        rev_ids += [tokenizer.pad_token_id] * padding_length
        rev_mask += [0] * padding_length

        # target
        if stage == "test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length - 2]
        target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
        #target_ids = tokenizer.convert_tokens_to_ids(target_tokens, vocab_kind='tgt') # trans
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] * len(target_ids)
        padding_length = args.max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length
        target_mask += [0] * padding_length

        token_length = len(source_ids)
        src_real_length = 1
        pdg_real_length = 1

        for j in range(1, token_length):
            if source_ids[j] == 0:
                # print(i, j, source_ids[i, j])
                src_real_length = j - 1
            if source_ids[j] == 2:
                # print(i, j, source_ids[i, j])
                pdg_real_length = j - src_real_length - 2
                break

        if example_index < 5:
            if stage == 'train':
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example.idx))

                logger.info("source_tokens: {}".format([x.replace('\u0120', '_') for x in source_tokens]))
                logger.info("source_ids: {}".format(' '.join(map(str, source_ids))))
                logger.info("source_mask: {}".format(' '.join(map(str, source_mask))))
                logger.info("source_length: {} pdg_length: {}".format(src_real_length,pdg_real_length))

                logger.info("target_tokens: {}".format([x.replace('\u0120', '_') for x in target_tokens]))
                logger.info("target_ids: {}".format(' '.join(map(str, target_ids))))
                logger.info("target_mask: {}".format(' '.join(map(str, target_mask))))


        features.append(
            InputFeatures(
                example_index,

                source_ids,
                dfs_ids,
                rev_ids,
                target_ids,

                source_mask,
                dfs_mask,
                rev_mask,
                target_mask,

                src_real_length,
                pdg_real_length
            )
        )
    return features


def set_seed(args):
    """set random seed."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters  
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type: e.g. roberta")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files")
    ## Other parameters
    parser.add_argument("--train_filename", default=None, type=str,
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_filename", default=None, type=str,
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_filename", default=None, type=str,
                        help="The test filename. Should contain the .jsonl files for this task.")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")

    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--vocab_size', type=int, default=50000,
                        help="maximum size of vocab")
    parser.add_argument('--hidden_size', type=int, default=512,
                        help="hidden_size for transformer encoder and decoder")
    parser.add_argument('--device', type=str, default='cpu',
                        help="device")
    # print arguments
    args = parser.parse_args()
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)
    logger.info(args)




    # Setup CUDA, GPU & distributed training
    """if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1"""

    device = torch.device(args.device if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 1
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
    args.device = device
    # Set seed
    set_seed(args)


    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,do_lower_case=args.do_lower_case)

    """
    tokenizer = Tokenizer(dev_file=args.dev_filename, train_file=args.train_filename)
    args.src_vocab_size = tokenizer.src_vocab_size
    args.tgt_vocab_size = tokenizer.tgt_vocab_size # trans
    """

    # budild model,model_class=roberta load codebert
    encoder = model_class.from_pretrained(args.model_name_or_path, config=config)
    #encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
    #encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

    decoder_layer1 = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    #decoder_layer1 = nn.TransformerDecoderLayer(d_model=512, nhead=8)
    decoder1 = nn.TransformerDecoder(decoder_layer1, num_layers=6)

    # decoder_layer2 = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    # decoder2 = nn.TransformerDecoder(decoder_layer2, num_layers=3)

    # decoder_layer3 = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    # decoder3 = nn.TransformerDecoder(decoder_layer3, num_layers=3)

    model = Seq2Seq(encoder=encoder, decoder1=decoder1, config=config, args=args,
                    beam_size=args.beam_size, max_length=args.max_target_length,
                    sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)
    if args.load_model_path is not None:
        logger.info("reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))

    model.to(device)
    if args.local_rank != -1:
        # Distributed training
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)

    if args.do_train:
        # Prepare training data loader
        train_examples = read_examples(args.train_filename)
        train_features = convert_examples_to_features(train_examples, tokenizer, args, stage='train')
        all_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
        all_source_mask = torch.tensor([f.source_mask for f in train_features], dtype=torch.long)

        all_dfs_ids = torch.tensor([f.dfs_ids for f in train_features], dtype=torch.long)
        all_dfs_mask = torch.tensor([f.dfs_mask for f in train_features], dtype=torch.long)

        all_rev_ids = torch.tensor([f.rev_ids for f in train_features], dtype=torch.long)
        all_rev_mask = torch.tensor([f.rev_mask for f in train_features], dtype=torch.long)

        all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long)
        all_target_mask = torch.tensor([f.target_mask for f in train_features], dtype=torch.long)

        all_source_real_length = torch.tensor([f.source_real_length for f in train_features], dtype=torch.long)
        all_pdg_real_length = torch.tensor([f.pdg_real_length for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_source_ids, all_source_mask, all_dfs_ids, all_dfs_mask,
                                   all_rev_ids, all_rev_mask, all_target_ids, all_target_mask,
                                   all_source_real_length, all_pdg_real_length)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                      batch_size=args.train_batch_size // args.gradient_accumulation_steps)

        num_train_optimization_steps = args.train_steps

        # Prepare optimizer and schedule (linear warmup and decay)
        if args.load_model_path is not None and args.do_train:
            last_output_dir2 = os.path.join(args.output_dir, 'checkpoint-last')
            output_model_file2 = os.path.join(last_output_dir2, "pytorch_model.bin")

            logger.info("  load optimizer, shceduler from path: {} ".format(output_model_file2))

            with open(output_model_file2 + 'optimizer_state_dict.pt', 'rb') as f:
                optimizer_state_dict = torch.load(f, map_location='cpu')
            with open(output_model_file2 + 'scheduler_obj.pt', 'rb') as f:
                scheduler_obj = torch.load(f, map_location='cpu')
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': args.weight_decay},
                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
            # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
            #                                            num_training_steps=num_train_optimization_steps)
            optimizer.load_state_dict(optimizer_state_dict)
            scheduler = scheduler_obj

        else:
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': args.weight_decay},
                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]
            logger.info("   initiate optimizer and shceduler")
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                        num_training_steps=num_train_optimization_steps)

        # Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num epoch = %d", num_train_optimization_steps * args.train_batch_size // len(train_examples))

        model.train()
        dev_dataset = {}
        nb_tr_examples, nb_tr_steps, tr_loss, global_step, best_bleu, best_loss = 0, 0, 0, 0, 0, 1e6
        bar = tqdm(range(num_train_optimization_steps), total=num_train_optimization_steps)
        train_dataloader = cycle(train_dataloader)
        eval_flag = True
        for step in bar:
            batch = next(train_dataloader)
            batch = tuple(t.to(device) for t in batch)

            source_ids, source_mask, dfs_ids, dfs_mask, rev_ids, rev_mask, target_ids, target_mask, source_real_length, pdg_real_length = batch

            loss, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                                   dfs_ids=dfs_ids, dfs_mask=dfs_mask,
                                   rev_ids=rev_ids, rev_mask=rev_mask,
                                   target_ids=target_ids, target_mask=target_mask,
                                   src_real_len=source_real_length,
                                   pdg_real_len=pdg_real_length)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            tr_loss += loss.item()
            train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
            bar.set_description("Training loss {}".format(train_loss))
            nb_tr_examples += source_ids.size(0)
            nb_tr_steps += 1
            loss.backward()

            if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                # Update parameters
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                eval_flag = True

            if args.do_eval and ((global_step + 1) % args.eval_steps == 0) and eval_flag:
                # Eval model with dev dataset
                logger.info("  Eval_start ")
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0
                eval_flag = False


                # save last checkpoint
                logger.info("  Training loss:{}  epoch:{}".format(train_loss, global_step * args.train_batch_size // len(train_examples)))
                logger.info("  save last checkpoint ")
                last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                if not os.path.exists(last_output_dir):
                    os.makedirs(last_output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)

                with open(output_model_file + 'optimizer_state_dict.pt', 'wb') as f:
                    torch.save(optimizer.state_dict(), f, pickle_module=dill)
                with open(output_model_file + 'scheduler_obj.pt', 'wb') as f:
                    torch.save(scheduler, f, pickle_module=dill)

                # Calculate bleu
                logger.info("  Calculate bleu ")
                if 'dev_bleu' in dev_dataset:
                    eval_examples, eval_data = dev_dataset['dev_bleu']
                else:
                    eval_examples = read_examples(args.dev_filename)
                    eval_examples = random.sample(eval_examples, min(1000, len(eval_examples)))
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='test')
                    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
                    all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)

                    all_dfs_ids = torch.tensor([f.dfs_ids for f in eval_features], dtype=torch.long)
                    all_dfs_mask = torch.tensor([f.dfs_mask for f in eval_features], dtype=torch.long)

                    all_rev_ids = torch.tensor([f.rev_ids for f in eval_features], dtype=torch.long)
                    all_rev_mask = torch.tensor([f.rev_mask for f in eval_features], dtype=torch.long)

                    eval_data = TensorDataset(all_source_ids, all_source_mask, all_dfs_ids, all_dfs_mask, all_rev_ids,
                                              all_rev_mask)
                    dev_dataset['dev_bleu'] = eval_examples, eval_data

                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                model.eval()
                p = []
                # cnt = 0

                # logger.info("  for batch in eval_dataloader: %d\n", len(eval_dataloader))
                for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc='Evaluation:'):
                    """cnt += 1
                    if cnt % 100 == 0:
                        logger.info("  for batch in eval_dataloader: %d/%d", cnt, len(eval_dataloader))"""
                    batch = tuple(t.to(device) for t in batch)
                    source_ids, source_mask, dfs_ids, dfs_mask, rev_ids, rev_mask = batch
                    with torch.no_grad():
                        preds = model(source_ids=source_ids, source_mask=source_mask, dfs_ids=dfs_ids,
                                      dfs_mask=dfs_mask, rev_ids=rev_ids, rev_mask=rev_mask)
                        for pred in preds:
                            t = pred[0].cpu().numpy()
                            t = list(t)
                            if 0 in t:
                                t = t[:t.index(0)]
                            text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                            p.append(text)
                model.train()
                predictions = []
                with open(os.path.join(args.output_dir, "dev.output"), 'w') as f, open(
                        os.path.join(args.output_dir, "dev.gold"), 'w') as f1:
                    # logger.info('  for ref, gold in zip(p, eval_examples):')  # p is predicted
                    for ref, gold in zip(p, eval_examples):
                        predictions.append(str(gold.idx) + '\t' + ref)
                        f.write(str(gold.idx) + '\t' + ref + '\n')
                        f1.write(str(gold.idx) + '\t' + gold.target + '\n')
                # logger.info("  (goldMap, predictionMap) = bleu.computeMaps(predictions, os.path.join(args.output_dir ")
                (goldMap, predictionMap) = bleu.computeMaps(predictions, os.path.join(args.output_dir, "dev.gold"))
                dev_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
                logger.info("  %s = %s " % ("bleu-4", str(dev_bleu)))
                logger.info("  " + "*" * 20)
                if dev_bleu > best_bleu:
                    logger.info("  Best bleu:%s", dev_bleu)
                    logger.info("  " + "*" * 20)
                    best_bleu = dev_bleu
                    # Save best checkpoint for best bleu
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                logger.info("  Eval_end ")

    if args.do_test:
        files = []
        if args.test_filename is not None:
            files.append(args.test_filename)
        for idx, file in enumerate(files):
            logger.info("Test file: {}".format(file))
            eval_examples = read_examples(file)
            eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='test')
            all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
            all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)

            all_dfs_ids = torch.tensor([f.dfs_ids for f in eval_features], dtype=torch.long)
            all_dfs_mask = torch.tensor([f.dfs_mask for f in eval_features], dtype=torch.long)

            all_rev_ids = torch.tensor([f.rev_ids for f in eval_features], dtype=torch.long)
            all_rev_mask = torch.tensor([f.rev_mask for f in eval_features], dtype=torch.long)

            eval_data = TensorDataset(all_source_ids, all_source_mask, all_dfs_ids, all_dfs_mask, all_rev_ids,
                                      all_rev_mask)

            # Calculate bleu
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

            model.eval()
            p = []  # List[str] ['return the value', ... , 'return a instance']
            for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
                batch = tuple(t.to(device) for t in batch)
                source_ids, source_mask, dfs_ids, dfs_mask, rev_ids, rev_mask = batch
                with torch.no_grad():
                    preds = model(source_ids=source_ids, source_mask=source_mask, dfs_ids=dfs_ids, dfs_mask=dfs_mask,
                                  rev_ids=rev_ids, rev_mask=rev_mask)
                    # print("===> preds", preds.size())
                    for pred in preds:
                        for pp in pred:
                            t = pp.cpu().numpy()
                            t = list(t)
                            if 0 in t:
                                t = t[:t.index(0)]
                            text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                            p.append(text)
            model.train()
            print("examples_num:", len(p))
            predictions = []
            with open(os.path.join(args.output_dir, "test_{}.output".format(str(idx))), 'w') as f:
                for ref in p:
                    # predictions.append(str(gold.idx)+'\t'+ref)
                    # f.write(str(gold.idx)+'\t'+ref+'\n')
                    f.write(ref + '\n')
                    # f1.write(str(gold.idx)+'\t'+gold.target+'\n')

            """(goldMap, predictionMap) = bleu.computeMaps(predictions,
                                                        os.path.join(args.output_dir, "test_{}.output".format(idx)))
            dev_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
            logger.info("  %s = %s " % ("bleu-4", str(dev_bleu)))
            logger.info("  " + "*" * 20)"""

            ref_dict = {}
            hyp_dict = {}
            for example in eval_examples:
                ref_dict.update({example.idx: [example.target]})
            for i, pre in enumerate(p):
                hyp_dict.update({i: [pre]})

            c_bleu, rouge_l, meteor = compute_score(references=ref_dict, hypotheses=hyp_dict)
            logger.info("  bleu:\t{}\trouge_l:\t{}\tmeteor:\t{}".format(c_bleu, rouge_l, meteor))


def compute_score(references: dict,
                  hypotheses: dict):  # hyptheses = {0: ['retrun treturn token2'], ..., 8703:['inserts a byte attay']}

    assert (sorted(references.keys()) == sorted(hypotheses.keys()))

    _, bleu, ind_bleu = corpus_bleu(hypotheses, references)

    # Compute ROUGE scores
    rouge_calculator = Rouge()
    rouge_l, ind_rouge = rouge_calculator.compute_score(references, hypotheses)

    # Compute METEOR scores

    try:
        meteor_calculator = Meteor()
        meteor, _ = meteor_calculator.compute_score(references, hypotheses)
    except:
        meteor = -1
    # meteor = -1
    return bleu, rouge_l, meteor


if __name__ == "__main__":
    with torch.cuda.device(0):
        main()
