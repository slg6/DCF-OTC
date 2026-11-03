from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors, Regex
import json
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def build_BPE(corpora_path):
    """构建BPE词汇表函数
    使用WordPiece算法构建网络流量数据的词汇表，确保词汇固定为4个字符
    
    参数:
        corpora_path: 语料库文件路径
    """
    num_count = 65541  # 词汇表大小
    # 特殊token: 'PAD':0,'UNK':1,'CLS':2,'SEP':3,'MASK':4
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    not_change_string_count = len(special_tokens)
    
    # 生成初始字典:将0-65535的数字转换为4位十六进制字符串作为key
    source_dictionary = {}
    i = 0
    while i < num_count:
        # 确保生成的是4位十六进制字符串
        temp_string = '{:04x}'.format(i)
        source_dictionary[temp_string] = i
        i += 1
    
    # 初始化WordPiece分词器
    tokenizer = Tokenizer(models.WordPiece(
        vocab=source_dictionary,
        unk_token="[UNK]",
        max_input_chars_per_word=4  # 限制每个词最大4个字符
    ))

    # 使用正则表达式匹配任意4个字符（除了换行符）
    four_char_pattern = Regex(r'.{4}')
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.BertPreTokenizer(),
        pre_tokenizers.Split(pattern=four_char_pattern, behavior="isolated")
    ])
    
    tokenizer.decoder = decoders.WordPiece()
    
    # 添加BERT风格的处理器
    tokenizer.post_processor = processors.BertProcessing(
        sep=("[SEP]", special_tokens.index("[SEP]")),
        cls=("[CLS]", special_tokens.index("[CLS]"))
    )

    # 初始化WordPiece训练器
    trainer = trainers.WordPieceTrainer(
        vocab_size=num_count,
        min_frequency=2,
        special_tokens=special_tokens
    )
    
    # 训练分词器
    tokenizer.train([corpora_path, corpora_path], trainer=trainer)
    
    # 过滤词汇表，只保留4个字符的词汇
    vocab = tokenizer.get_vocab()
    # filtered_vocab = {k: v for k, v in vocab.items() if len(k) == 4 or k in special_tokens}
    filtered_vocab = {k: v for k, v in vocab.items()}
    
    # 如果过滤后词汇不足，补充一些4字符词汇
    if len(filtered_vocab) < num_count:
        missing = num_count - len(filtered_vocab)
        print(f"补充 {missing} 个4字符词汇以达到目标词汇量")
        i = 0
        while len(filtered_vocab) < num_count:
            candidate = f"temp{i:03d}"  # 生成临时4字符词汇
            if candidate not in filtered_vocab:
                filtered_vocab[candidate] = max(filtered_vocab.values()) + 1
            i += 1
    
    # 更新分词器的词汇表
    tokenizer.model = models.WordPiece(
        vocab=filtered_vocab,
        unk_token="[UNK]",
        max_input_chars_per_word=4
    )
    
    # 保存训练好的分词器
    tokenizer.save("wordpiece_4chars.tokenizer.json", pretty=True)
    return 0

def build_vocab(vocab_path):
    """构建词汇表文件，只包含4个字符的词汇
    
    参数:
        vocab_path: 词汇表输出路径
    """
    with open("wordpiece_4chars.tokenizer.json", 'r') as f:
        vocab_json = json.load(f)
    
    # 提取并过滤词汇，只保留4字符的词汇和特殊标记
    special_tokens = ["[PAD]", "[SEP]", "[CLS]", "[UNK]", "[MASK]"]
    vocab_entries = []
    
    # 先添加特殊标记
    # for token in special_tokens:
    #     if token in vocab_json['model']['vocab']:
    #         vocab_entries.append(token)
    
    # 再添加4字符词汇，按ID排序
    sorted_items = sorted(vocab_json['model']['vocab'].items(), key=lambda x: x[1])
    for item, _ in sorted_items:
        # if len(item) == 4 and item not in special_tokens and item not in vocab_entries:
        vocab_entries.append(item)
    
    # 写入词汇表文件
    with open(vocab_path, 'w') as f:
        for word in vocab_entries:
            f.write(word + "\n")
    
    print(f"生成的词汇表包含 {len(vocab_entries)} 个词汇")
    return 0

# 执行构建过程
# build_BPE('direction_bpe_corpus_tcp_udp.txt')
build_vocab("vocab_packet_4chars.txt")
    