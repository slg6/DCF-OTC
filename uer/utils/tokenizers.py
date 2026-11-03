# -*- encoding:utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from uer.utils.constants import *
from uer.utils.vocab import Vocab
import collections
import unicodedata
import six


class Tokenizer(object):

    def __init__(self, args, is_src=True):
        self.vocab = None
        self.sp_model = None
        if is_src == True:
            spm_model_path = args.spm_model_path
            vocab_path = args.vocab_path
        else:
            spm_model_path = args.tgt_spm_model_path
            vocab_path = args.tgt_vocab_path

        if spm_model_path:
            try:
                import sentencepiece as spm
            except ImportError:
                raise ImportError("You need to install SentencePiece to use XLNetTokenizer: https://github.com/google/sentencepiece"
                                                    "pip install sentencepiece")
            self.sp_model = spm.SentencePieceProcessor()
            self.sp_model.Load(spm_model_path)
            self.vocab = {self.sp_model.IdToPiece(i): i for i
                                        in range(self.sp_model.GetPieceSize())}
        else:
            self.vocab = Vocab()
            self.vocab.load(vocab_path, is_quiet=True)  #加载词
            self.vocab = self.vocab.w2i
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def tokenize(self, text):
        raise NotImplementedError

    def convert_tokens_to_ids(self, tokens):    #把tokens转换为ids
        if self.sp_model:
            return [self.sp_model.PieceToId(
                    printable_text(token)) for token in tokens]
        else:
            return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        if self.sp_model:
            return [self.sp_model.IdToPiece(id_) for id_ in ids]
        else:
            return convert_by_vocab(self.inv_vocab, ids)


class CharTokenizer(Tokenizer):
        
    def __init__(self, args, is_src=True):
        super().__init__(args, is_src)

    def tokenize(self, text, use_vocab=True):
        if use_vocab:
            return [token if token in self.vocab else "[UNK]" for token in list(text.strip())]
        else:
            return [token for token in list(text.strip())]


class SpaceTokenizer(Tokenizer):
     
    def __init__(self, args, is_src=True):
        super().__init__(args, is_src)

    def tokenize(self, text, use_vocab=True):
        if use_vocab:
            return [token if token in self.vocab else "[UNK]" for token in text.strip().split(" ")]
        else:
            return [token for token in text.strip().split(" ")]


SPIECE_UNDERLINE = u"▁".encode("utf-8")


def preprocess_text(inputs, remove_space=True, lower=False):
    """preprocess data by removing extra space and normalize data."""
    outputs = inputs
    if remove_space:
        outputs = " ".join(inputs.strip().split())

    if six.PY2 and isinstance(outputs, str):
        try:
            outputs = six.ensure_text(outputs, "utf-8")
        except UnicodeDecodeError:
            outputs = six.ensure_text(outputs, "latin-1")

    outputs = unicodedata.normalize("NFKD", outputs)
    outputs = "".join([c for c in outputs if not unicodedata.combining(c)])
    if lower:
        outputs = outputs.lower()

    return outputs


def encode_pieces(sp_model, text, return_unicode=True, sample=False):
    """turn sentences into word pieces."""

    if six.PY2 and isinstance(text, six.text_type):
        text = six.ensure_binary(text, "utf-8")

    if not sample:
        pieces = sp_model.EncodeAsPieces(text)
    else:
        pieces = sp_model.SampleEncodeAsPieces(text, 64, 0.1)
    new_pieces = []
    for piece in pieces:
        piece = printable_text(piece)
        if len(piece) > 1 and piece[-1] == "," and piece[-2].isdigit():
            cur_pieces = sp_model.EncodeAsPieces(
                    six.ensure_binary(piece[:-1]).replace(SPIECE_UNDERLINE, b""))
            if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
                if len(cur_pieces[0]) == 1:
                    cur_pieces = cur_pieces[1:]
                else:
                    cur_pieces[0] = cur_pieces[0][1:]
            cur_pieces.append(piece[-1])
            new_pieces.extend(cur_pieces)
        else:
            new_pieces.append(piece)

    # note(zhiliny): convert back to unicode for py2
    if six.PY2 and return_unicode:
        ret_pieces = []
        for piece in new_pieces:
            if isinstance(piece, str):
                piece = six.ensure_text(piece, "utf-8")
            ret_pieces.append(piece)
        new_pieces = ret_pieces

    return new_pieces


def encode_ids(sp_model, text, sample=False):
    pieces = encode_pieces(sp_model, text, return_unicode=False, sample=sample)
    ids = [sp_model.PieceToId(piece) for piece in pieces]
    return ids


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return six.ensure_text(text, "utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return six.ensure_text(text, "utf-8", "ignore")
        elif isinstance(text, six.text_type):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return six.ensure_text(text, "utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, six.text_type):
            return six.ensure_binary(text, "utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def convert_by_vocab(vocab, items):     #把tokens转换为ids
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        output.append(vocab[item])
    return output


def convert_tokens_to_ids(vocab, tokens):
    return convert_by_vocab(vocab, tokens)


def convert_ids_to_tokens(inv_vocab, ids):
    return convert_by_vocab(inv_vocab, ids)


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class BertTokenizer(Tokenizer):
    """Runs end-to-end tokenziation."""

    def __init__(self, args, is_src=True, do_lower_case=True):
        super().__init__(args, is_src)
        if not args.spm_model_path:
            self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)  #负责基础的文本预处理和分词，转小写、标点符号分割、清理无效字符...
            self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab) #执行词片段(WordPiece)分词，使用贪婪的最长匹配优先算法

    def tokenize(self, text):   #将输入文本转换为 token 序列
        if self.sp_model:       # 使用SentencePiece模型(如果有)
            split_tokens = encode_pieces(self.sp_model, text, return_unicode=False)
        else:           #使用基本分词器和词片段分词器
            split_tokens = []
            for token in self.basic_tokenizer.tokenize(text):   #token 是由 basic_tokenizer 分词后得到的基本单元 BasicTokenizer 的主要处理步骤：清理无效字符、按空格分割
                for sub_token in self.wordpiece_tokenizer.tokenize(token):
                    split_tokens.append(sub_token)

        return split_tokens #['[UNK]', '[UNK]', '[UNK]', '4500', '0000', '003c', '3c4b', '4b19', '1940', '4000', '0040', '4006', '066c', '6c06', '06ec', 'ec79', '7949', '491d', '1d7e', '7e52', '525d', '5db3', 'b3d7', 'd75d', '5d9b', '9b', '##2c', '2c49', '4931', '319c', '9c7a', '7a00', '0000', '0000', '0000', '00a0', 'a002', '0272', '7210', '1032', '32c7', 'c700', '0000', '0002', '0204', '0405', '05b4', 'b404', '0402', '0208', '080a', '0a6f', '6f8a', '8acd', 'cde4', 'e400', '0000', '0000', '0000', ...]


class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=True):
        """Constructs a BasicTokenizer.
        Args:
            do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = convert_to_unicode(text)
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        text = self._tokenize_chinese_chars(text)

        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    """Runs WordPiece tokenziation."""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
        self.vocab = vocab  #vocab = {'[PAD]': 0, '[SEP]': 1, '[CLS]': 2, '[UNK]': 3, '[MASK]': 4, '0': 5, '1': 6, '2': 7, '3': 8, '4': 9, '5': 10, '6': 11, '7': 12, '8': 13, '9': 14, 'a': 15, 'b': 16, 'c': 17, 'd': 18, 'e': 19, 'f': 20, '##6': 21, '##1': 22, '##7': 23, '##b': 24, '##5': 25, '##e': 26, '##a': 27, '##c': 28, '##3': 29, '##f': 30, '##4': 31, '##2': 32, '##9': 33, '##8': 34, '##0': 35, '##d': 36, '##00': 37, '00': 38, '000': 39, '##01': 40, '0000': 41, '##08': 42, '##40': 43, '08': 44, '01': 45, '##83': 46, '##45': 47, '##ac': 48, '##50': 49, '##3c': 50, '##e1': 51, '##4d': 52, '##09': 53, '##36': 54, '##fe': 55, '##f0': 56, '##dc': 57, '##bb': 58, ...}
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word    #用于限制单个词的最大字符数，此处最多为4

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.
        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.
        For example:
            input = "unaffable"
            output = ["un", "##aff", "##able"]
        Args:
            text: A single token or whitespace separated tokens. This should have
                already been passed through `BasicTokenizer.
        Returns:
            A list of wordpiece tokens.
        """
        """贪婪的最长匹配优先算法
        示例:
            输入 = "unaffable"
            输出 = ["un", "##aff", "##able"]
        """
        text = convert_to_unicode(text)

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:   #如果没有之间匹配就进行分割
                        substr = "##" + six.ensure_str(substr)
                    if substr in self.vocab:        #如果找到就不再分割
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically control characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat in ("Cc", "Cf"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False
