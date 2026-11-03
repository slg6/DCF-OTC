import itertools
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def generate_hex_4char_vocab(vocab_save_path="vocab_packet_4chars_all.txt"):
    # 1. 定义十六进制基础字符集（16个可选字符）
    hex_chars = "0123456789abcdef"  # 若需小写可改为"0123456789abcdef"，需与字段格式化规则一致
    vocab = ["[PAD]", "[SEP]", "[CLS]", "[UNK]", "[MASK]"]
    # 2. 生成所有四字符组合（16^4 = 65536种）
    # itertools.product生成笛卡尔积，等价于4层循环遍历每个位置的字符
    for chars in itertools.product(hex_chars, repeat=4):
        token = "".join(chars)  # 拼接为四字符token（如"0000"、"0001"、...、"FFFF"）
        vocab.append(token)
    
    # 3. 验证词表大小（确保为65536）
    assert len(vocab) == 65541, f"词表大小错误：实际{len(vocab)}个，预期65536个"
    
    # 4. 保存词表（格式：索引\ttoken，方便后续Tokenizer调用）
    with open(vocab_save_path, "w", encoding="utf-8") as f:
        for idx, token in enumerate(vocab):
            f.write(f"{token}\n")
    
    print(f"词表生成完成！共{len(vocab)}个token，已保存至{vocab_save_path}")
    return vocab

# 执行生成
if __name__ == "__main__":
    generate_hex_4char_vocab()
    