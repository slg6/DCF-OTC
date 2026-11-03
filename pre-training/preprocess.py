#!/usr/bin/python3
#-*- coding:utf-8 -*-
import argparse
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
sys.path.append(os.getcwd())
import six
from packaging import version
from uer.utils.data import *
from uer.utils import *


assert version.parse(six.__version__) >= version.parse("1.12.0")


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--corpus_path", type=str, default="vocab_gen/direction_bpe_corpus_tcp_udp.txt",
                        help="Path of the corpus for pretraining.") #预训练语料库的路径
    parser.add_argument("--vocab_path", default="vocab_gen/vocab_packet_4chars_all.txt", type=str, 
                        help="Path of the vocabulary file.")    #词汇表文件路径
    parser.add_argument("--spm_model_path", default=None, type=str,
                        help="Path of the sentence piece model.")
    parser.add_argument("--tgt_vocab_path", default=None, type=str,
                        help="Path of the target vocabulary file.")
    parser.add_argument("--tgt_spm_model_path", default=None, type=str,
                        help="Path of the target sentence piece model.")
    parser.add_argument("--dataset_path", type=str, default="pre-training/dataset/dataset.pt",
                        help="Path of the preprocessed dataset.")   #预处理后数据集的保存路径

    # Preprocess options.
    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                        help="Specify the tokenizer." 
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Space tokenizer segments sentences into words according to space."
                             )  #指定分词器类型
    parser.add_argument("--tgt_tokenizer", choices=["bert", "char", "space"], default="bert",
                        help="Specify the tokenizer.")
    
    #训练时将数据集分割为多少份并行处理
    parser.add_argument("--processes_num", type=int, default=30,
                        help="Split the whole dataset into `processes_num` parts, "
                             "and each part is fed to a single process in training step.")
    
    #训练目标类型
    parser.add_argument("--target", choices=["bert", "bertflow","bertpackets","lm", "mlm", "bilm", "albert", "seq2seq", "t5", "cls", "prefixlm"], default="bertpackets",
                        help="The training dataset target.")
    parser.add_argument("--docs_buffer_size", type=int, default=100000, #default=100000
                        help="The buffer size of documents in memory, specific to targets that require negative sampling.")
    
    #序列长度
    parser.add_argument("--seq_length", type=int, default=221, help="Sequence length of instances.") 
    parser.add_argument("--num_packet", type=int, default=20, help="Sequence length of instances.") 
    parser.add_argument("--tgt_seq_length", type=int, default=128, help="Target sequence length of instances.")
    parser.add_argument("--dup_factor", type=int, default=5,
                        help="Duplicate instances multiple times.") #多次复制实例。
    parser.add_argument("--short_seq_prob", type=float, default=0.1,
                        help="Probability of truncating sequence."
                             "The larger value, the higher probability of using short (truncated) sequence.")
    parser.add_argument("--full_sentences", action="store_true", help="Full sentences.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")

    # Masking options.
    parser.add_argument("--dynamic_masking", action="store_true", help="Dynamic masking.")
    parser.add_argument("--whole_word_masking", action="store_true", help="Whole word masking.")
    parser.add_argument("--span_masking", action="store_true", help="Span masking.")
    parser.add_argument("--span_geo_prob", type=float, default=0.2,
                        help="Hyperparameter of geometric distribution for span masking.")
    parser.add_argument("--span_max_length", type=int, default=10,
                        help="Max length for span masking.")

    args = parser.parse_args()

    # Dynamic masking.
    # 如果使用动态掩码，将重复因子设为1
    if args.dynamic_masking:
        args.dup_factor = 1

    # Build tokenizer.
    #构建分词器
    tokenizer = str2tokenizer[args.tokenizer](args) #BertTokenizer(args)    args.tokenizer:default="bert"
    if args.target == "seq2seq":
        args.tgt_tokenizer = str2tokenizer[args.tgt_tokenizer](args, False)

    # Build and save dataset.
    #构建并保存数据集
    dataset = str2dataset[args.target](args, tokenizer.vocab, tokenizer)#BertFlowDataset(args, tokenizer.vocab, tokenizer)
    '''
    词汇表（vocab.txt 或 encryptd_vocab.txt）是分词器的核心组件，它定义了所有可能的 token 及其对应的 ID
    BertTokenizer 是基于词汇表来对输入文本进行分词和编码的。
    对于输入字符串 abcd，分词器会按照以下逻辑进行分割：优先匹配最长的 token
    '''
    dataset.build_and_save(args.processes_num,split_by_flow=True)
    '''
    seq_length 是一个重要的预处理参数，设定了模型输入序列的最大长度，在代码中默认值为128。它的作用主要体现在以下几个方面：
        1. 序列截断和填充
        当输入序列长度超过 seq_length，会被截断
        当输入序列长度小于 seq_length，会用 PAD token 填充
        确保所有输入序列具有相同的长度，便于批处理
    '''


if __name__ == "__main__":
    main()