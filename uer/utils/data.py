import os
import random
import pickle
import torch
from multiprocessing import Pool
from uer.utils.constants import *
from uer.utils.tokenizers import *
from uer.utils.misc import count_lines
from uer.utils.seed import set_seed
from uer.utils.fieldNegativeGenerator import FieldNegativeGenerator as FNG
#slg
from copy import deepcopy

def mask_seq(src, tokenizer, whole_word_masking, span_masking, span_geo_prob, span_max_length):
    vocab = tokenizer.vocab     #从分词器中获取词汇表。

    for i in range(len(src) - 1, -1, -1):      # 跳过尾部的填充（PAD）：
        if src[i] != PAD_ID:
            break
    src_no_pad = src[:i + 1]
    tokens_index, src_no_pad = create_index(src_no_pad, tokenizer, whole_word_masking, span_masking, span_geo_prob, span_max_length)
    if len(src_no_pad) < len(src):
        src = src_no_pad + (len(src) - len(src_no_pad)) * [PAD_ID]
    else:
        src = src_no_pad

    random.shuffle(tokens_index)
    num_to_predict = max(1, int(round(len(src_no_pad) * 0.15))) #num_to_predict:掩码的token数量  0.15:默认掩码比例为 15%（BERT 的标准设置）。  至少掩码一个 token（max(1, ...)）。
    tgt_mlm = []
    for index_set in tokens_index:
        if len(tgt_mlm) >= num_to_predict:
            break
        if whole_word_masking:
            i = index_set[0]
            mask_len = index_set[1]
            if len(tgt_mlm) + mask_len > num_to_predict:
                continue

            for j in range(mask_len):
                token = src[i + j]
                tgt_mlm.append((i + j, token))
                prob = random.random()
                if prob < 0.8:
                    src[i + j] = vocab.get(MASK_TOKEN)
                elif prob < 0.9:
                    while True:
                        rdi = random.randint(1, len(vocab) - 1)
                        if rdi not in [vocab.get(CLS_TOKEN), vocab.get(SEP_TOKEN), vocab.get(MASK_TOKEN), PAD_ID]:
                            break
                    src[i + j] = rdi
        elif span_masking:
            i = index_set[0]
            span_len = index_set[1]
            if len(tgt_mlm) + span_len > num_to_predict:
                continue

            for j in range(span_len):
                token = src[i + j]
                tgt_mlm.append((i + j, token))
            prob = random.random()
            if prob < 0.8:
                for j in range(span_len):
                    src[i + j] = vocab.get(MASK_TOKEN)
            elif prob < 0.9:
                for j in range(span_len):
                    while True:
                        rdi = random.randint(1, len(vocab) - 1)
                        if rdi not in [vocab.get(CLS_TOKEN), vocab.get(SEP_TOKEN), vocab.get(MASK_TOKEN), PAD_ID]:
                            break
                    src[i + j] = rdi
        else:
            i = index_set[0]
            token = src[i]
            tgt_mlm.append((i, token))  #被掩码的位置，被掩码前的值
            prob = random.random()
            if prob < 0.8:  #80% 的概率用掩码 token 替换。
                src[i] = vocab.get(MASK_TOKEN)
            elif prob < 0.9: #10% 的概率用随机 token 替换。    
                while True:
                    rdi = random.randint(1, len(vocab) - 1)
                    if rdi not in [vocab.get(CLS_TOKEN), vocab.get(SEP_TOKEN), vocab.get(MASK_TOKEN), PAD_ID]:
                        break
                src[i] = rdi        #10% 的概率保持原 token 不变。
    tgt_mlm = sorted(tgt_mlm, key=lambda x: x[0])
    return src, tgt_mlm


def create_index(src, tokenizer, whole_word_masking, span_masking, span_geo_prob, span_max_length):
    tokens_index = []   #用于存储掩码索引。
    span_end_position = -1  #记录跨度掩码的结束位置。
    vocab = tokenizer.vocab     #从分词器中获取词汇表。
    if whole_word_masking:  #false
        src_wwm = []
        src_length = len(src)
        has_cls, has_sep = False, False
        if src[0] == vocab.get(CLS_TOKEN):
            src = src[1:]
            has_cls = True
        if src[-1] == vocab.get(SEP_TOKEN):
            src = src[:-1]
            has_sep = True
        sentence = "".join(tokenizer.convert_ids_to_tokens(src)).replace('[UNK]', '').replace('##', '')
        import jieba
        wordlist = jieba.cut(sentence)
        if has_cls:
            src_wwm += [vocab.get(CLS_TOKEN)]
        for word in wordlist:
            position = len(src_wwm)
            src_wwm += tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
            if len(src_wwm) < src_length:
                tokens_index.append([position, len(src_wwm)-position])
        if has_sep:
            src_wwm += [vocab.get(SEP_TOKEN)]
        if len(src_wwm) > src_length:
            src = src_wwm[:src_length]
        else:
            src = src_wwm
    else:
        for (i, token) in enumerate(src):
            if token == vocab.get(CLS_TOKEN) or token == vocab.get(SEP_TOKEN) or token == PAD_ID:
                continue
            if not span_masking:    #如果未启用跨度掩码
                tokens_index.append([i])
            else:
                if i < span_end_position:
                    continue
                span_len = get_span_len(span_max_length, span_geo_prob)
                span_end_position = i + span_len
                if span_end_position > len(src):
                    span_len = len(src) - i
                tokens_index.append([i, span_len])
    return tokens_index, src


def get_span_len(max_span_len, p):
    geo_prob_cum = [0.0]
    geo_prob = 1.0
    for i in range(max_span_len + 1):
        if i == 0:
            continue
        if i == 1:
            geo_prob *= p
            geo_prob_cum.append(geo_prob_cum[-1] + geo_prob)
        else:
            geo_prob *= (1 - p)
            geo_prob_cum.append(geo_prob_cum[-1] + geo_prob)

    prob = geo_prob_cum[-1] * random.random()
    for i in range(len(geo_prob_cum) - 1):
        if prob >= geo_prob_cum[i] and prob < geo_prob_cum[i + 1]:
            current_span_len = i + 1
    return current_span_len


def merge_dataset(dataset_path, workers_num):
    # Merge datasets.
    #slg
    # 创建临时目录
    temp_dir = "temp_debug"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    #slg
    dataset_writer = open(dataset_path, "wb")
    for i in range(workers_num):
        # if i == 1:
        #     continue
        # print(f"合并{i}个文件")
        # tmp_dataset_reader = open("/mnt/data/zgm/ET-BERT/datasets/temp/dataset-tmp-" + str(i) + ".pt", "rb") #源代码
        #slg
        tmp_path = os.path.join(temp_dir, f"dataset-tmp-{i}.pt")
        tmp_dataset_reader = open(tmp_path, "rb")
        #slg
        while True:
            tmp_data = tmp_dataset_reader.read(2**22) 
            if tmp_data:
                dataset_writer.write(tmp_data)
            else:
                break
        tmp_dataset_reader.close()
        # os.remove("/mnt/data/zgm/ET-BERT/datasets/temp/dataset-tmp-" + str(i) + ".pt") #源代码
        #slg
        os.remove(tmp_path) #不清理
        #slg
    dataset_writer.close()
    #slg
    # 清理临时目录
    os.rmdir(temp_dir)        #不清理
    #slg

def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
    """ truncate sequence pair to specific length """
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break
        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b  #比较 tokens_a 和 tokens_b 的长度，选择较长的序列进行截断。
        if random.random() < 0.5: 
            del trunc_tokens[0] #如果随机数小于 0.5，则删除序列的第一个元素（del trunc_tokens[0]）。
        else:
            trunc_tokens.pop()  #否则，删除序列的最后一个元素（trunc_tokens.pop()）。

def record_flow_start(corpus_path):
    starts = []
    i = 0
    with open(corpus_path, mode="r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line[:2] == "||":
                starts.append(i)
            i+=1
            # if len(starts) == 100:
            #     break
    starts.append(i)
    return starts, i

class Dataset(object):
    def __init__(self, args, vocab, tokenizer):
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.corpus_path = args.corpus_path
        self.dataset_path = args.dataset_path
        self.seq_length = args.seq_length
        self.seed = args.seed
        self.dynamic_masking = args.dynamic_masking
        self.whole_word_masking = args.whole_word_masking
        self.span_masking = args.span_masking
        self.span_geo_prob = args.span_geo_prob
        self.span_max_length = args.span_max_length
        self.docs_buffer_size = args.docs_buffer_size
        self.dup_factor = args.dup_factor

    def build_and_save(self, workers_num, split_by_flow=False):
        """
        Build dataset from the given corpus.
        Start workers_num processes and each process deals with a part of data.
        主要功能是将原始语料处理成训练数据并保存
        """
        # lines_num = count_lines(self.corpus_path)
        if split_by_flow:
            starts, lines_num = record_flow_start(self.corpus_path)
        else:
            lines_num = count_lines(self.corpus_path)
        print("Starting %d workers for building datasets ... " % workers_num)
        assert (workers_num >= 1)
        if workers_num == 1:
            self.worker(0, 0, lines_num)
        else:
            pool = Pool(workers_num)
            # if split_by_flow:
            #     starts = record_flow_start(self.corpus_path)        #预训练语料库中 每个流的起始行号
            # current_index = 0
            for i in range(workers_num):
                if split_by_flow:
                    # start = starts[current_index]
                    # for j in range(len(starts))[current_index:]:
                    #     if starts[j]-starts[current_index]>perburst:
                    #         current_index = j
                    #         break
                    # if i==workers_num-1:
                    #     current_index = len(starts)-1
                    # end = starts[current_index]
                    start = starts[i*(len(starts)-1)//workers_num]
                    end = starts[(i+1)*(len(starts)-1)//workers_num]    #一个进程分一部分语料
                else:
                    start = i * lines_num // workers_num
                    end = (i + 1) * lines_num // workers_num
                pool.apply_async(func=self.worker, args=[i, start, end])        #每个进程处理一部分语料
            pool.close()
            #slg
            # for _ in tqdm.tqdm(range(workers_num), desc="Processing", unit="worker"):
            pool.join()
            #slg
            # i = 1
            # start = i * lines_num // workers_num
            # end = (i + 1) * lines_num // workers_num
            # self.worker(i, start, end)
            #slg
        # Merge datasets.
        merge_dataset(self.dataset_path, workers_num)   #合并保存，并行处理的数据集
        

    def worker(self, proc_id, start, end):
        raise NotImplementedError()


class DataLoader(object):
    def __init__(self, args, dataset_path, batch_size, proc_id, proc_num, shuffle=False):
        self.tokenizer = args.tokenizer
        self.batch_size = batch_size
        self.instances_buffer_size = args.instances_buffer_size #默认为25600
        self.proc_id = proc_id
        self.proc_num = proc_num
        self.shuffle = shuffle
        self.dataset_reader = open(dataset_path, "rb")
        self.read_count = 0
        self.start = 0
        self.end = 0
        self.buffer = []
        self.vocab = args.vocab
        self.whole_word_masking = args.whole_word_masking
        self.span_masking = args.span_masking
        self.span_geo_prob = args.span_geo_prob
        self.span_max_length = args.span_max_length
    # def _fill_buf(self):
    #     try:
    #         self.buffer = []
    #         while True:
    #             instance = pickle.load(self.dataset_reader)
    #             self.read_count += 1
    #     except EOFError:        #instance = pickle.load(self.dataset_reader) 加载完所有实例后继续加载会报错EOFError
    #         # Reach file end.
    #         print(self.read_count)
    #         self.dataset_reader.seek(0)     #文件从头加载

    def _fill_buf(self):
        try:
            self.buffer = []
            while True:
                instance = pickle.load(self.dataset_reader)   #共有63504950个实例
                self.read_count += 1
                if (self.read_count - 1) % self.proc_num == self.proc_id: 
                    self.buffer.append(instance)
                    if len(self.buffer) >= self.instances_buffer_size:
                        break
        except EOFError:        #instance = pickle.load(self.dataset_reader) 加载完所有实例后继续加载会报错EOFError
            # Reach file end.
            self.dataset_reader.seek(0)     #文件从头加载

        if self.shuffle:
            random.shuffle(self.buffer)
        self.start = 0
        self.end = len(self.buffer)

    def _empty(self):
        return self.start >= self.end

    def __del__(self):
        self.dataset_reader.close()


class BertDataset(Dataset):
    """
    Construct dataset for MLM and NSP tasks from the given corpus.
    Each document consists of multiple sentences,
    and each sentence occupies a single line.
    Documents in corpus must be separated by empty lines.
    """

    def __init__(self, args, vocab, tokenizer):
        super(BertDataset, self).__init__(args, vocab, tokenizer)
        self.short_seq_prob = args.short_seq_prob

    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        #slg
        # 创建临时目录
        temp_dir = "temp"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            
        # 使用相对路径
        tmp_path = os.path.join(temp_dir, f"dataset-tmp-{proc_id}.pt")
        dataset_writer = open(tmp_path, "wb")
        #slg
        docs_buffer = []
        document = []
        pos = 0
        # dataset_writer = open("/mnt/data/zgm/ET-BERT/datasets/temp/dataset-tmp-" + str(proc_id) + ".pt", "wb") #源代码
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                f.readline()
                pos += 1
            while True:
                line = f.readline()
                pos += 1
                if pos >= end:
                    if len(docs_buffer) > 0:
                        instances = self.build_instances(docs_buffer)
                        for instance in instances:
                            pickle.dump(instance, dataset_writer)
                    break

                if not line.strip(): 
                    if len(document) >= 1:
                        docs_buffer.append(document)
                    document = []
                    if len(docs_buffer) == self.docs_buffer_size: 
                        print("Worker {}: Docs buffer is full, build instances...".format(proc_id))
                        # Build instances from documents.
                        instances = self.build_instances(docs_buffer)
                        # Save instances.
                        for instance in instances: 
                            pickle.dump(instance, dataset_writer)
                        # Clear buffer.
                        docs_buffer = []
                    continue
                if line[:2] == "||":
                    line = line[2:]
                sentence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(line)) 
                if len(sentence) > 0:
                    document.append(sentence)

        dataset_writer.close()

    def build_instances(self, all_documents):
        instances = []
        for _ in range(self.dup_factor):        #多次复制实例
            for doc_index in range(len(all_documents)):
                instances.extend(self.create_ins_from_doc(all_documents, doc_index))
                '''
                all_documents = 
                [
                    [       #docs_buffer
                        [       #document
                            [senttece]
                            [senttece]
                        ]

                        ...
                    ]
                    
                    ...
                ]
                '''
        return instances

    def create_ins_from_doc(self, all_documents, document_index):
        document = all_documents[document_index]
        max_num_tokens = self.seq_length - 3
        target_seq_length = max_num_tokens
        if random.random() < self.short_seq_prob:
            target_seq_length = random.randint(2, max_num_tokens)
        instances = []
        current_chunk = []
        current_length = 0
        i = 0
        while i < len(document):
            segment = document[i]
            current_chunk.append(segment)
            current_length += len(segment)
            if i == len(document) - 1 or current_length >= target_seq_length:
                if current_chunk:
                    a_end = 1
                    if len(current_chunk) >= 2: 
                        a_end = random.randint(1, len(current_chunk) - 1)

                    tokens_a = [] #seg 1
                    for j in range(a_end):
                        tokens_a.extend(current_chunk[j])

                    tokens_b = [] ##seg 2
                    is_random_next = 0

                    if len(current_chunk) == 1 or random.random() < 0.5: 
                        is_random_next = 1
                        target_b_length = target_seq_length - len(tokens_a)

                        for _ in range(10):
                            random_document_index = random.randint(0, len(all_documents) - 1)
                            if random_document_index != document_index:
                                break

                        random_document = all_documents[random_document_index]
                        random_start = random.randint(0, len(random_document) - 1)
                        for j in range(random_start, len(random_document)):
                            tokens_b.extend(random_document[j])
                            if len(tokens_b) >= target_b_length:
                                break

                        num_unused_segments = len(current_chunk) - a_end 
                        i -= num_unused_segments

                    else:
                        is_random_next = 0
                        for j in range(a_end, len(current_chunk)):
                            tokens_b.extend(current_chunk[j])

                    truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)
  
                    src = []
                    src.append(self.vocab.get(CLS_TOKEN))
                    src.extend(tokens_a)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos = [len(src)]
                    src.extend(tokens_b)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos.append(len(src))

                    while len(src) != self.seq_length:
                        src.append(PAD_ID)

                    if not self.dynamic_masking:
                        src, tgt_mlm = mask_seq(src, self.tokenizer, self.whole_word_masking, self.span_masking, self.span_geo_prob, self.span_max_length)
                        instance = (src, tgt_mlm, is_random_next, seg_pos)
                    else:
                        instance = (src, is_random_next, seg_pos)

                    instances.append(instance)
                current_chunk = []
                current_length = 0
            i += 1
        return instances

class BertFlowDataset(Dataset):
    """
    Construct dataset for MLM and MIX tasks from the given corpus.
    Each document consists of multiple paragraphs,
    Each paragraph consists of multiple sentences,
    and each sentence occupies a single line.
    Paragraphs in corpus must be separated by empty lines.
    Documents in corpus must be separated by empty lines.
    """

    def __init__(self, args, vocab, tokenizer):
        super(BertFlowDataset, self).__init__(args, vocab, tokenizer)
        self.short_seq_prob = args.short_seq_prob

    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        print(start,end)        #处理的起始行，终止行
        set_seed(self.seed)
        #slg
        # 创建临时目录
        temp_dir = "temp"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            
        # 使用相对路径
        tmp_path = os.path.join(temp_dir, f"dataset-tmp-{proc_id}.pt")
        dataset_writer = open(tmp_path, "wb")
        #slg
        flow_buffer = []    #存储处理后的流量数据。
        flow_proto = [] #存储协议信息。
        docs_buffer = []    #存储当前文档的段落数据。
        document = []   #存储当前段落的句子数据。
        pos = 0 #当前处理的行号。
        # dataset_writer = open("/mnt/data/zgm/ET-BERT/datasets/temp/dataset-tmp-" + str(proc_id) + ".pt", "wb") #源代码
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            try:
                #with open(self.corpus_path[:-4]+"_extra.txt", mode="r", encoding="utf-8") as fe:
                while pos < start:
                    f.readline()
                    pos += 1
                while True:
                    line = f.readline()
                    if pos==start and line[:2] != "||":
                        print("not flow start...")
                    pos += 1
                    if pos > end:
                        if len(docs_buffer) >= 1:   #len(docs_buffer[i]) == 2,是突发的两部分  docs_buffer是这个进程要处理的所有的突发
                            flow_buffer.append(docs_buffer)
                            
                        if len(flow_buffer) > 0:        #len(flow_buffer) 是处理部分含有多少条流量，  len(flow_buffer[i])  是i条流量中含有多少对突发
                            try:
                                instances = self.build_instances(flow_buffer,flow_proto) #生成实例
                            except Exception as e:
                                print("has error2...", len(flow_buffer), len(flow_proto), e)
                            for instance in instances:
                                pickle.dump(instance, dataset_writer)
                        break

                    if not line.strip(): 
                        if len(document) >= 1:
                            docs_buffer.append(document)
                        document = []
                        continue
                    if line[:2] == "||" or not line: 
                        if len(docs_buffer) >= 1:
                            flow_buffer.append(docs_buffer)
                        docs_buffer = []
                    
                        flow_buffer_size = 0
                        for d in flow_buffer:
                            flow_buffer_size += len(d)
                        if flow_buffer_size > self.docs_buffer_size or pos>=end: 
                            print("Worker %d is building instances ... " % proc_id, len(flow_buffer),len(flow_proto))
                            try:
                                instances = self.build_instances(flow_buffer,flow_proto)    #生成实例
                            except:
                                print("has error1...")
                            print("Worker {} has {} instances. ".format(proc_id,len(instances)))
                            # Save instances.
                            for instance in instances: 
                                pickle.dump(instance, dataset_writer)
                            flow_buffer = []
                            flow_proto = []
                        if pos>=end or not line:
                            break
                        line = line[2:]
                        if line[0]=="4":
                            if "bigram" in self.corpus_path:
                                if line[42:44] == "06":
                                    flow_proto.append(0)
                                elif line[42:44] == "11":
                                    flow_proto.append(1)
                                else:
                                    print("not tcp or udp, ",line[42:44])
                            else:
                                if line[22:24] == "06":
                                    flow_proto.append(0)
                                elif line[22:24] == "11":
                                    flow_proto.append(1)
                        else:
                            print("find Ipv6!!")

                    sentence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(line)) 
                    if len(sentence) > 0:       #是一个列表，每个元素是一个token id  [162, 41, 2749, 16286, 15450, 4181, 144, 358, 697, 4652, 13465, 10670, 7748, 3225, 3457, 29975, 13592, 24465, 27572, 50279, 20248, 7626, 244, 3924, 14261, 29182, 6813, 3442, 41]
                        document.append(sentence)

            except:
                print("has error...")
        print("Worker %d finished... " % proc_id)
        dataset_writer.close()

    def build_instances(self, all_documents, flow_proto):
        assert len(all_documents) == len(flow_proto)
        instances = []
        for _ in range(self.dup_factor):    #多次复制实例 dup_factor默认为5
            for doc_index in range(len(all_documents)):
                # if doc_index%500==0:
                #     print(doc_index)
                instances.extend(self.create_ins_from_doc(all_documents, doc_index, flow_proto))
        return instances
    def create_ins_from_doc(self, all_documents,document_index, flow_proto):
        document = all_documents[document_index]    #一个流，一个元素为一对列表（一对突发）document[0] = [[162, 41, 2749, 16286, 15450, 4181, 144, 358, 697, 4652, 13465, 10670, 7748, 3225, 3457, 29975, 13592, 24465, 27572, ...], [41, 3522, 13864, 9818, 560, 28869, 6883, 3442, 41, 503, 5584, 2703, 6903, 21926, 2532, 706, 440, 611, 41, ...]]
        max_num_tokens = self.seq_length - 3
        target_seq_length = max_num_tokens
        if random.random() < self.short_seq_prob: 
            target_seq_length = random.randint(2, max_num_tokens)   #target_seq_length设置为2-max_num_tokens之间的一个随机数
        instances = []
        i=0
        while i < len(document): 
            rnd = random.random()
            if i==len(document)-1 or rnd < 0.4: 
                a_end = 1
                if len(document[i]) >= 2: 
                    a_end = random.randint(1, len(document[i]) - 1) #此处len(document[i])必须大于等于1

                tokens_a = [] #seg 1
                for j in range(a_end):
                    tokens_a.extend(document[i][j])

                tokens_b = [] ##seg 2
                for j in range(a_end, len(document[i])):
                    tokens_b.extend(document[i][j])
                
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)
                if random.random()<0.5:  #A1 A2
                    label = 0 #
                    src = []
                    src.append(self.vocab.get(CLS_TOKEN))
                    src.extend(tokens_a)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos = [len(src)]
                    src.extend(tokens_b)
                    src.append(self.vocab.get(SEP_TOKEN))   #src:CLS, token_A, SEP, token_B, SEP
                    seg_pos.append(len(src))    #seg_pos: len(CLS+token_A), len(CLS+token_A+SEP_token_B+SEP)
                else: #A2 A1
                    label = 1
                    src = []
                    src.append(self.vocab.get(CLS_TOKEN))
                    src.extend(tokens_b)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos = [len(src)]
                    src.extend(tokens_a)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos.append(len(src))
            elif rnd < 0.6: 
                tokens_a = []
                for j in range(len(document[i])):
                    tokens_a.extend(document[i][j])
                next_burst_max_length = target_seq_length - len(tokens_a)

                for _ in range(20):
                    random_document_index = random.randint(0, len(all_documents) - 1)
                    if random_document_index != document_index:
                        break

                random_document = all_documents[random_document_index] 
                random_start = random.randint(0, len(random_document) - 1)
                burst_ind_end = random_start+1 #len(random_document) 
                tokens_b = []
                for burst_ind in range(random_start, burst_ind_end): 
                    for j in range(len(random_document[burst_ind])): 
                        tokens_b.extend(random_document[burst_ind][j])
                        if len(tokens_b) >= next_burst_max_length:
                            break
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)
                if random_document_index == document_index:
                    if random_start>=i:
                        label = 3 
                    else:
                        label = 4
                else:
                    label = 2 # A1A2 Z1Z2
                src = []
                src.append(self.vocab.get(CLS_TOKEN))
                src.extend(tokens_a)
                src.append(self.vocab.get(SEP_TOKEN))
                seg_pos = [len(src)]
                src.extend(tokens_b)
                src.append(self.vocab.get(SEP_TOKEN))
                seg_pos.append(len(src))
            else:
                tokens_a = []
                for j in range(len(document[i])):
                    tokens_a.extend(document[i][j])
                i+=1
                tokens_b = []
                for j in range(len(document[i])):
                    tokens_b.extend(document[i][j])
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)
                if random.random()<0.5:  #A1A2 B1B2
                    label = 3 
                    src = []
                    src.append(self.vocab.get(CLS_TOKEN))
                    src.extend(tokens_a)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos = [len(src)]
                    src.extend(tokens_b)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos.append(len(src))
                else: #B1B2 A1A2
                    label = 4
                    src = []
                    src.append(self.vocab.get(CLS_TOKEN))
                    src.extend(tokens_b)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos = [len(src)]
                    src.extend(tokens_a)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos.append(len(src))
            while len(src) != self.seq_length:
                src.append(PAD_ID)      #填充
            if not self.dynamic_masking:
                src, tgt_mlm = mask_seq(src, self.tokenizer, self.whole_word_masking, self.span_masking, self.span_geo_prob, self.span_max_length)  #掩码后的src  掩码的信息
                instance = (src, tgt_mlm, label, seg_pos)
            else:
                instance = (src, label, seg_pos)
            instance += (flow_proto[document_index],)
            instances.append(instance)
            i+=1 
    
        return instances

class BertPacketsDataset(Dataset):
    """
    Construct dataset for MLM and MIX tasks from the given corpus.
    Each document consists of multiple paragraphs,
    Each paragraph consists of multiple sentences,
    and each sentence occupies a single line.
    Paragraphs in corpus must be separated by empty lines.
    Documents in corpus must be separated by empty lines.
    """

    def __init__(self, args, vocab, tokenizer):
        super(BertPacketsDataset, self).__init__(args, vocab, tokenizer)
        self.short_seq_prob = args.short_seq_prob
        self.num_packet = args.num_packet

    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        print(start,end)        #处理的起始行，终止行
        set_seed(self.seed)
        #slg
        # 创建临时目录
        temp_dir = "temp_debug"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            
        # 使用相对路径
        tmp_path = os.path.join(temp_dir, f"dataset-tmp-{proc_id}.pt")
        dataset_writer = open(tmp_path, "wb")
        #slg
        flow_buffer = []    #所有的信息，一个元素是一个flow列表，即一个docs_buffer。
        flow_proto = [] #存储协议信息。
        docs_buffer = []    #flow，一个元素是一个packet 字段token列表
        # document = []   #packet，
        pos = 0 #当前处理的行号。
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            # try:
            #with open(self.corpus_path[:-4]+"_extra.txt", mode="r", encoding="utf-8") as fe:
            while pos < start:   # 跳过初始行，直到达到起始位置
                f.readline()
                pos += 1
            while True:      # 主处理循环
                line = f.readline()     #读的带 \n  读完文件后，后面读的内容均为空字符串''
                if pos==start and line[:2] != "||":
                    print("not flow start...")
                pos += 1
                if pos > end:   # 如果超过结束位置，生成实例，并退出循环
                    if len(docs_buffer) >= 1:   #处理剩余数据
                        flow_buffer.append(docs_buffer)
                        
                    if len(flow_buffer) > 0:
                        # try:
                        instances = self.build_instances(flow_buffer,flow_proto)  # 生成实例
                        # except Exception as e:
                        #     print("has error2...", len(flow_buffer), len(flow_proto), e)
                        for instance in instances:
                            pickle.dump(instance, dataset_writer)   #将instance写入dataset_writer
                    break

                # if not line.strip():    # 如果遇到空行，表示段落结束    line.strip() 方法会移除字符串 line 的首尾空白字符（包括空格、制表符、换行符等）。
                #     if len(document) >= 1: 
                #         docs_buffer.append(document)
                #     document = []
                    # continue
                if line[:2] == "||" or not line:        # 如果遇到流量标记行（以 "||" 开头）或 空行结束     对于字符串来说，空字符串 "" 是假值，而任何非空字符串都是真值（truthy value）。
                    if len(docs_buffer) >= 1:        # 如果有未处理的段落，添加到流量缓冲区
                        flow_buffer.append(docs_buffer)
                    docs_buffer = []
                
                    flow_buffer_size = 0    #总包数
                    for d in flow_buffer:
                        flow_buffer_size += len(d)  
                    if flow_buffer_size > self.docs_buffer_size or pos>=end: #一个流超过docs_buffer_size个字符，开始生成实例，默认为100000
                        print("Worker %d is building instances ... " % proc_id, len(flow_buffer),len(flow_proto))
                        # try:
                        instances = self.build_instances(flow_buffer,flow_proto)    #生成实例
                        # except:
                        #     print("has error1...")
                        print("Worker {} has {} instances. ".format(proc_id,len(instances)))
                        # Save instances.
                        for instance in instances: 
                            pickle.dump(instance, dataset_writer)
                        flow_buffer = []
                        flow_proto = []
                    if pos>=end or not line:  #只有文件结束才为空行，其余的行至少有一个 换行符 \n 
                        break
                    # line = line[2:]
                    line = line[2:]     #能走到这里说明，line的前两字符一定为||，否则在上面会break
                    # if line[13]=="4":
                    if line[32:34] == "06": # TCP协议
                        flow_proto.append(0)
                    elif line[32:34] == "11":   # UDP协议
                        flow_proto.append(1)
                    else:
                        print("not udp or tcp")

                # line_len = int(len(line) / 2) - int(len(line) / 2) % 5
                # line1 = line[:line_len]
                # line2 = line[line_len:]
                # sentence1 = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(line1)) 
                # sentence2 = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(line2)) 
                # if len(sentence1) > 0 and len(sentence2) > 0:
                #     docs_buffer.append([sentence1, sentence2])
                # sentence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(line))  #将句子进行分词，如果 line 是 "Hello, world!"，分词器可能会将其分割为 ["Hello", ",", "world", "!"]。
                # if len(sentence) > 0:
                #     docs_buffer.append(sentence)   #分词后的数据包
                sentence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(line)) 
                if len(sentence) > 0:       #是一个列表，每个元素是一个token id  [162, 41, 2749, 16286, 15450, 4181, 144, 358, 697, 4652, 13465, 10670, 7748, 3225, 3457, 29975, 13592, 24465, 27572, 50279, 20248, 7626, 244, 3924, 14261, 29182, 6813, 3442, 41]
                    docs_buffer.append(sentence)
            # except:
            #     print("has error...")
        print("Worker %d finished... " % proc_id)
        dataset_writer.close()

    def build_instances(self, all_documents, flow_proto):
        assert len(all_documents) == len(flow_proto),f"文档数 {len(all_documents)} 与流原型数 {len(flow_proto)} 不一致！"
        instances = []
        for _ in range(self.dup_factor):    #多次复制实例   
                                            # 每次复制时，代码中的随机选择（如 random.random() 和 random.randint()）会导致不同的 tokens_a 和 tokens_b 的组合。
                                            # 这种随机性可以模拟不同的输入顺序和上下文，使模型能够学习到更广泛的模式。
            # print("dup_factor:", _)       动态掩码时，dup_factor被设置为1
            for doc_index in range(len(all_documents)): 
                # if doc_index%500==0:
                #     print(doc_index)
                # print("build_instances:", doc_index)
                instances.extend(self.create_ins_from_doc(all_documents, doc_index, flow_proto))
        # print("build_instances:", len(instances))
        return instances
    def shuffle_packets(self, packets, prob_local_reverse=0.4, prob_swap_k=0.4, prob_interval_swap=0.2):
        """
        对包头序列进行多样化局部乱序，生成负样本
        :param original_packets: 原始包头列表（长度为20）
        :param prob_local_reverse: 局部逆序的概率
        :param prob_swap_k: 随机交换k个包的概率
        :param prob_interval_swap: 间隔置换的概率
        :return: 打乱后的包头列表
        """
        n = len(packets)
        if n == 1:
            return packets, 0
        rand_prob = random.random()
        lable = 0
        if rand_prob < prob_local_reverse:  # 局部逆序
            min_segment = 2
            max_segment = min(n//2, 5)  
            if max_segment < min_segment:
                max_segment = min_segment
            start = random.randint(0, n - max_segment)
            length = random.randint(min_segment, max_segment)
            end = start + length
            packets[start:end] = packets[start:end][::-1] #逆序
            lable = 1
        if rand_prob < prob_local_reverse + prob_swap_k:    #随机交换k个不相邻的包
            if n <= 3:
                return packets, lable
            max_k = min((n+2)//3, 5)
            min_k = 2
            k = random.randint(min_k, max_k)   #交换k个包（2，5）个
            indices = []
            while len(indices) < k:
                idx = random.randint(0, n - 1)
                # if n <= 4:
                #     if idx not in indices:
                #         indices.append(idx)
                # else:
                if all(abs(idx - i) > 1 for i in indices) and idx not in indices:   #all如果所有元素都为真，则返回True
                    indices.append(idx)
            # 打乱选中的索引顺序并交换
            shuffled_indices = sorted(indices, key=lambda x: random.random())   #sorted 将列表 indices 随机打乱顺序，但不改变原列表
            while shuffled_indices==indices:
                shuffled_indices = sorted(indices, key=lambda x: random.random())
            for i, j in zip(indices, shuffled_indices):
                packets[i], packets[j] = packets[j], packets[i]
            lable = 2
        else:
            if n >= 3:
                if random.random() < 0.5:
                    positions = list(range(1, n, 2))  # 打乱奇数位置包
                else:
                    positions = list(range(0, n, 2)) # 打乱偶数位置包
            else:
                positions = list(range(n))  # n=2时为[0,1]
            if len(positions) >= 2:
                shuffled_positions = sorted(positions, key=lambda x: random.random())
                while shuffled_positions == positions:      #打乱到二者不一样为止
                    shuffled_positions = sorted(positions, key=lambda x: random.random())
                for i, j in zip(positions, shuffled_positions):
                    packets[i], packets[j] = packets[j], packets[i]
            else:
                i, j = random.sample(range(n), 2)
                packets[i], packets[j] = packets[j], packets[i]
            lable = 3
        
        return packets, lable

    def create_ins_from_doc(self, all_documents,document_index, flow_proto):
        '''
        all_documents: 包含所有文档的列表，每个元素是一个流量（flow）的列表。
        document_index: 当前处理的文档索引。
        flow_proto: 存储协议信息的列表
        '''
        document = deepcopy(all_documents[document_index]) #一个flow流  30个包
        num_packet = self.num_packet    #每个流处理20个包
        # max_num_tokens = self.seq_length - 21    # 最大token数量(421)，减去[CLS]、[SEP]和其他特殊token  400
        if num_packet > len(document):  #如果该流的包数量小于20，则只处理该流的包数
            num_packet = len(document)
        #通过随机生成较短序列，能让模型在预训练阶段接触到更接近真实分布的长度变化，避免模型仅适应固定长度（如最大序列长度）的输入。
        if random.random() < self.short_seq_prob:       #short_seq_prob  default=0.1 以一定概率生成较短的序列  
            num_packet = random.randint(1, num_packet)
        document = document[:num_packet]   #截断，只保留前num_packet个包
        instances = []  #实例列表

        label = 0
        if random.random() < 0.6: #0.6的概率，打乱包序
            document, label = self.shuffle_packets(document[:num_packet])  #对前num_packet个包进行乱序
        else:       #剩下0.4的概率，保持顺序
            label = 0
        
        field_mask_list = []
        
        if random.random() < 0.5:   #0.5的概率，修改字段值
            fng = FNG()     #用于生成无效字段
            for packet_idex in range(len(document)):
                if random.random() < 0.6:   #每个包被修改的概率为0.6
                    proto_ = "tcp" if flow_proto[document_index]==0 else "udp"
                    packet = document[packet_idex]
                    modify_count = random.randint(1, int(len(packet)*0.25)) #修改字段的数量，至少修改一个，最多修改25%的字段
                    fng.generate_negative_sample(packet, proto_, modify_count, field_mask_list, packet_idex)  #document[packet_idex] 在函数内被修改
        src = []
        seg_pos = []
        src.append(self.vocab.get(CLS_TOKEN))
        seg_pos.append(1)   #CLS
        for packet_index in range(num_packet):  #【CLS 包1 SEP 包2 SEP ...... 包n SEP】
            src.extend(document[packet_index])
            src.append(self.vocab.get(SEP_TOKEN))
            ip_seg = [packet_index*2+1] * 7 #ip头7字段
            seg_pos.extend(ip_seg)
            # if flow_proto[document_index] == 0:
            #     transport_seg = [packet_index*2 + 1] * 9 #tcp头8字段 + SEP
            # else:
            #     transport_seg = [packet_index * 10 + 2] * 9 #udp头8字段 + SEP
            transport_seg = [packet_index*2 + 2] * 4
            seg_pos.extend(transport_seg)
        while len(src) != self.seq_length:
            src.append(PAD_ID)      #PAD_ID:0
            seg_pos.append(PAD_ID)
        if not self.dynamic_masking:
            #进行掩码
            src, tgt_mlm = mask_seq(src, self.tokenizer, self.whole_word_masking, self.span_masking, self.span_geo_prob, self.span_max_length)  #掩码后的src 掩码信息
            instance = (src, tgt_mlm, label, field_mask_list, seg_pos)    #掩码、改序后的src 掩码信息 数据是否连续 字段值是否有效 数据包的位置 段位置
        else:
            instance = (src, label, field_mask_list, seg_pos)
        instance += (flow_proto[document_index],)       #协议
        instances.append(instance)
        # i+=1 
    
        return instances

class BertDataLoader(DataLoader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            tgt_mlm = []
            is_next = []
            seg = []

            masked_words_num = 0

            for ins in instances:
                if len(ins) == 4: 
                    src.append(ins[0])
                    masked_words_num += len(ins[1])
                    tgt_mlm.append([0] * len(ins[0]))
                    for mask in ins[1]:
                        tgt_mlm[-1][mask[0]] = mask[1]
                    is_next.append(ins[2])
                    seg.append([1] * ins[3][0] + [2] * (ins[3][1] - ins[3][0]) + [PAD_ID] * (len(ins[0]) - ins[3][1]))
                else:
                    src_single, tgt_mlm_single = mask_seq(ins[0], self.tokenizer, self.whole_word_masking, self.span_masking, self.span_geo_prob, self.span_max_length)
                    masked_words_num += len(tgt_mlm_single)
                    src.append(src_single)
                    tgt_mlm.append([0] * len(ins[0]))
                    for mask in tgt_mlm_single:
                        tgt_mlm[-1][mask[0]] = mask[1]
                    is_next.append(ins[1])
                    seg.append([1] * ins[2][0] + [2] * (ins[2][1] - ins[2][0]) + [PAD_ID] * (len(ins[0]) - ins[2][1]))

            if masked_words_num == 0:
                continue

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt_mlm), \
                torch.LongTensor(is_next), \
                torch.LongTensor(seg)

class BertFlowDataLoader(DataLoader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]   #一次取出batch_size的数据量

            self.start += self.batch_size

            src = []    #源序列（输入文本）。 [CLS] 开头，以 [SEP] 结尾，中间可能包含 [MASK] 标记。
            tgt_mlm = []    #掩码语言模型（MLM）的目标标签。  格式为 [(position, target_token_id), ...]，表示被掩码的位置及其目标 Token ID。
            is_next = []    #下一句预测（NSP）的目标标签。  0 表示句子对是连续的。  1 表示句子对是随机拼接的。
            seg = []    #段嵌入  表示不同段落的起始位置，格式为 [segment1_end, segment2_end, ...]
            proto = []  #协议信息  整数，表示当前实例的协议类型（例如，0 表示 TCP，1 表示 UDP）。
            '''
            具体实例：
            instance = (
                [101, 7592, 103, 102, 2023, 103, 1034, 102, 0, 0, ...],  # src
                [(3, 2129), (6, 2003)],                              # 3，6被掩码  掩码前值为2129，2003
                0,                                                    #  0 表示句子对是连续的。  1 表示句子对是随机拼接的。  修改成数据包位置
                [4, 8],                                              # Segment A 结束于索引 4，Segment B 结束于索引 8
                0                                                     # 协议为TCP
            )
            '''
            masked_words_num = 0

            for ins in instances:
                if len(ins) == 5: 
                    src.append(ins[0])  #源序列
                    masked_words_num += len(ins[1]) #被掩码的单词数量
                    tgt_mlm.append([0] * len(ins[0]))   #初始化tgt列表
                    for mask in ins[1]:
                        tgt_mlm[-1][mask[0]] = mask[1]
                    is_next.append(ins[2])
                    seg.append([1] * ins[3][0] + [2] * (ins[3][1] - ins[3][0]) + [PAD_ID] * (len(ins[0]) - ins[3][1]))
                    proto.append(ins[4])
                else:
                    src_single, tgt_mlm_single = mask_seq(ins[0], self.tokenizer, self.whole_word_masking, self.span_masking, self.span_geo_prob, self.span_max_length)
                    masked_words_num += len(tgt_mlm_single)
                    src.append(src_single)
                    tgt_mlm.append([0] * len(ins[0]))
                    for mask in tgt_mlm_single:
                        tgt_mlm[-1][mask[0]] = mask[1]
                    is_next.append(ins[1])
                    seg.append([1] * ins[2][0] + [2] * (ins[2][1] - ins[2][0]) + [PAD_ID] * (len(ins[0]) - ins[2][1]))
                    proto.append(ins[3])
            if masked_words_num == 0:
                continue

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt_mlm), \
                torch.LongTensor(is_next), \
                torch.LongTensor(seg), \
                torch.LongTensor(proto)

class BertPacketsDataLoader(DataLoader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()        #每个进程平均加载实例，每个进程最多加载25600个实例。
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]   #一次取出batch_size的数据量
            self.start += self.batch_size
            src = []    #源序列（输入文本）。 [CLS] 开头，以 [SEP] 结尾，中间可能包含 [MASK] 标记。
            tgt_mlm = []    #掩码语言模型（MLM）的目标标签。  格式为 [(position, target_token_id), ...]，表示被掩码的位置及其目标 Token ID。
            label = []    #下一句预测（NSP）的目标标签。  0 表示句子对是连续的。  1 表示局部逆序。 2表示不相邻交换k个包 3表示间隔置换
            field_mask = []    #字段替换  表示被随机修改的字段，格式为 [(field_index, field_value), ...]
            seg = []   #段嵌入  表示不同段，格式为 [0,0,0,0,1,1,1,1,10,10,10,10,11,11,11,11...]  每个包的字段值相同
            proto = []  #协议信息  整数，表示当前实例的协议类型（例如，0 表示 TCP，1 表示 UDP）。
            '''
            具体实例：
            instance = (
                [101, 7592, 103, 102, 2023, 103, 1034, 102, 0, 0, ...],  # src
                [(3, 2129), (6, 2003)],                              # 3，6被掩码  掩码前值为2129，2003
                0,                                                    #  0 表示句子对是连续的。  1 表示句子对是随机拼接的。  修改成数据包位置
                [(5, 2120), (21, 9)],                              # 5，21被随机替换  替换前值为2120，9
                [0,0,0,0,1,1,1,1,10,10,10,10,11,11,11,11...]  # seg
                0                                                     # 协议为TCP
            )
            '''
            masked_words_num = 0
            modified_field_num = 0
            for ins in instances:
                if len(ins) == 6: 
                    src.append(ins[0])  #源序列
                    masked_words_num += len(ins[1]) #被掩码的单词数量
                    tgt_mlm.append([0] * len(ins[0]))   #初始化tgt列表
                    for mask in ins[1]:
                        tgt_mlm[-1][mask[0]] = mask[1]
                    label.append(ins[2])
                    modified_field_num += len(ins[3])
                    field_mask.append([0] * len(ins[0]))
                    for fmodifide in ins[3]:
                        field_mask[-1][fmodifide[0]] = fmodifide[1]
                    seg.append(ins[4])
                    proto.append(ins[5])
                else:
                    src_single, tgt_mlm_single = mask_seq(ins[0], self.tokenizer, self.whole_word_masking, self.span_masking, self.span_geo_prob, self.span_max_length)
                    masked_words_num += len(tgt_mlm_single)
                    src.append(src_single)
                    tgt_mlm.append([0] * len(ins[0]))
                    for mask in tgt_mlm_single:
                        tgt_mlm[-1][mask[0]] = mask[1]
                    label.append(ins[1])
                    modified_field_num += len(ins[2])
                    field_mask.append([0] * len(ins[0]))
                    for fmodifide in ins[2]:
                        field_mask[-1][fmodifide[0]] = fmodifide[1]
                    seg.append(ins[3])
                    proto.append(ins[4])
            if masked_words_num == 0:
                continue

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt_mlm), \
                torch.LongTensor(label), \
                torch.LongTensor(field_mask), \
                torch.LongTensor(seg), \
                torch.LongTensor(proto)

class MlmDataset(Dataset):
    def __init__(self, args, vocab, tokenizer):
        super(MlmDataset, self).__init__(args, vocab, tokenizer)
        self.full_sentences = args.full_sentences

    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        dataset_writer = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        docs_buffer = []
        for _ in range(self.dup_factor):
            pos = 0
            with open(self.corpus_path, mode="r", encoding="utf-8") as f:
                while pos < start:
                    f.readline()
                    pos += 1
                while True:
                    line = f.readline()
                    pos += 1

                    document = [self.vocab.get(CLS_TOKEN)] + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(line)) + [self.vocab.get(SEP_TOKEN)]

                    if self.full_sentences:
                        if len(document) > 0:
                            docs_buffer.append(document)
                        if len(docs_buffer) == self.docs_buffer_size:
                            # Build instances from documents.
                            all_documents = self.concatenate_docs(docs_buffer)
                            instances = self.build_instances(all_documents)
                            # Save instances.
                            for instance in instances:
                                pickle.dump(instance, dataset_writer)
                            # Clear buffer.
                            docs_buffer = []
                        if pos >= end:
                            if len(docs_buffer) > 0:
                                all_documents = self.concatenate_docs(docs_buffer)
                                instances = self.build_instances(all_documents)
                                # Save instances.
                                for instance in instances:
                                    pickle.dump(instance, dataset_writer)
                            break
                    else:
                        if len(document) > 0:
                            instances = self.build_instances(document)
                            # Save instances.
                            for instance in instances:
                                pickle.dump(instance, dataset_writer)

                    if pos >= end:
                        break

        dataset_writer.close()

    def concatenate_docs(self, docs_buffer):
        all_documents = []
        for i in range(len(docs_buffer)):
            all_documents += docs_buffer[i]
        return all_documents

    def build_instances(self, all_documents):
        instances = []
        instances_num = len(all_documents) // self.seq_length
        for i in range(instances_num):
            src = all_documents[i * self.seq_length: (i + 1) * self.seq_length]
            seg_pos = [len(src)]

            if not self.dynamic_masking:
                src, tgt = mask_seq(src, self.tokenizer, self.whole_word_masking, self.span_masking, self.span_geo_prob, self.span_max_length)
                instance = (src, tgt, seg_pos)
            else:
                instance = (src, seg_pos)

            instances.append(instance)

        src = all_documents[instances_num * self.seq_length:]
        seg_pos = [len(src)]

        while len(src) != self.seq_length:
            src.append(PAD_ID)

        if not self.dynamic_masking:
            src, tgt = mask_seq(src, self.tokenizer, self.whole_word_masking, self.span_masking, self.span_geo_prob, self.span_max_length)
            instance = (src, tgt, seg_pos)
        else:
            instance = (src, seg_pos)

        instances.append(instance)
        return instances


class MlmDataLoader(DataLoader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            tgt = []
            seg = []

            masked_words_num = 0

            for ins in instances:
                if len(ins) == 3:
                    src.append(ins[0])
                    masked_words_num += len(ins[1])
                    tgt.append([0] * len(ins[0]))
                    for mask in ins[1]:
                        tgt[-1][mask[0]] = mask[1]
                    seg.append([1] * ins[2][0] + [PAD_ID] * (len(ins[0]) - ins[2][0]))
                else:
                    src_single, tgt_single = mask_seq(ins[0], self.tokenizer, self.whole_word_masking, self.span_masking, self.span_geo_prob, self.span_max_length)
                    masked_words_num += len(tgt_single)
                    src.append(src_single)
                    tgt.append([0] * len(ins[0]))
                    for mask in tgt_single:
                        tgt[-1][mask[0]] = mask[1]
                    seg.append([1] * ins[1][0] + [PAD_ID] * (len(ins[0]) - ins[1][0]))

            if masked_words_num == 0:
                continue

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt), \
                torch.LongTensor(seg)


class AlbertDataset(Dataset):
    """
    Construct dataset for MLM and SOP tasks from the given corpus.
    Each document consists of multiple sentences,
    and each sentence occupies a single line.
    Documents in corpus must be separated by empty lines.
    """

    def __init__(self, args, vocab, tokenizer):
        super(AlbertDataset, self).__init__(args, vocab, tokenizer)
        self.short_seq_prob = args.short_seq_prob

    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        document = []
        dataset_writer = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        for _ in range(self.dup_factor):
            pos = 0
            with open(self.corpus_path, mode="r", encoding="utf-8") as f:
                while pos < start:
                    f.readline()
                    pos += 1
                while True:
                    line = f.readline()
                    pos += 1
                    if not line.strip():
                        if len(document) >= 1:
                            instances = self.build_instances(document)
                            for instance in instances:
                                pickle.dump(instance, dataset_writer)
                        document = []
                    sentence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(line))
                    if len(sentence) > 0:
                        document.append(sentence)
                    if pos >= end - 1:
                        if len(document) >= 1:
                            instances = self.build_instances(document)
                            for instance in instances:
                                pickle.dump(instance, dataset_writer)
                        break
        dataset_writer.close()

    def build_instances(self, document):
        instances = []
        instances.extend(self.create_ins_from_doc(document))
        return instances

    def create_ins_from_doc(self, document):
        max_num_tokens = self.seq_length - 3
        target_seq_length = max_num_tokens
        if random.random() < self.short_seq_prob:
            target_seq_length = random.randint(2, max_num_tokens)
        instances = []
        current_chunk = []
        current_length = 0
        i = 0
        while i < len(document):
            segment = document[i]
            current_chunk.append(segment)
            current_length += len(segment)
            if i == len(document) - 1 or current_length >= target_seq_length:
                if current_chunk:
                    a_end = 1
                    if len(current_chunk) >= 2:
                        a_end = random.randint(1, len(current_chunk) - 1)

                    tokens_a = []
                    for j in range(a_end):
                        tokens_a.extend(current_chunk[j])

                    tokens_b = []
                    is_wrong_order = 0
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])

                    if random.random() < 0.5:
                        is_wrong_order = 1
                        tmp = tokens_a
                        tokens_a = tokens_b
                        tokens_b = tmp

                    truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)

                    src = []
                    src.append(self.vocab.get(CLS_TOKEN))
                    src.extend(tokens_a)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos = [len(src)]
                    src.extend(tokens_b)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos.append(len(src))

                    while len(src) != self.seq_length:
                        src.append(PAD_ID)

                    if not self.dynamic_masking:
                        src, tgt_mlm = mask_seq(src, self.tokenizer, self.whole_word_masking, self.span_masking, self.span_geo_prob, self.span_max_length)
                        instance = (src, tgt_mlm, is_wrong_order, seg_pos)
                    else:
                        instance = (src, is_wrong_order, seg_pos)

                    instances.append(instance)
                current_chunk = []
                current_length = 0
            i += 1
        return instances


class AlbertDataLoader(BertDataLoader):
    '''
    AlbertDataLoader can reuse the code of BertDataLoader.
    '''
    pass


class LmDataset(Dataset):
    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        dataset_writer = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        pos = 0
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                f.readline()
                pos += 1
            while True:
                line = f.readline()
                pos += 1

                document = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(line))
                document = [self.vocab.get(CLS_TOKEN)] + document + [self.vocab.get(SEP_TOKEN)]

                instances_num = len(document) // (self.seq_length + 1)
                for i in range(instances_num):
                    src = document[i * (self.seq_length + 1): (i + 1) * (self.seq_length + 1)]
                    seg_pos = self.seq_length
                    pickle.dump((src, seg_pos), dataset_writer)

                src = document[instances_num * (self.seq_length + 1):]
                if len(src) > 0:
                    seg_pos = len(src)
                    while len(src) != self.seq_length + 1:
                        src.append(PAD_ID)
                    pickle.dump((src, seg_pos), dataset_writer)

                if pos >= end:
                    break

        dataset_writer.close()


class LmDataLoader(DataLoader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            tgt = []
            seg = []

            for ins in instances:
                src.append(ins[0][:-1])
                tgt.append(ins[0][1:])
                if ins[1] == len(ins[0]):
                    seg.append([1] * (ins[1] - 1))
                else:
                    seg.append([1] * ins[1] + [PAD_ID] * (len(ins[0]) - 1 - ins[1]))

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt), \
                torch.LongTensor(seg)


class BilmDataset(Dataset):
    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        dataset_writer = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        pos = 0
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                f.readline()
                pos += 1
            while True:
                line = f.readline()
                pos += 1

                document = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(line))

                instances_num = len(document) // self.seq_length
                for i in range(instances_num):
                    src = document[i * self.seq_length: (i + 1) * self.seq_length]
                    tgt_forward = src[1:] + [self.vocab.get(SEP_TOKEN)]
                    tgt_backward = [self.vocab.get(CLS_TOKEN)] + src[:-1]
                    seg = [1] * len(src)
                    pickle.dump((src, tgt_forward, tgt_backward, seg), dataset_writer)

                src = document[instances_num * self.seq_length:]
                if len(src) < 1:
                    continue
                tgt_forward = src[1:] + [self.vocab.get(SEP_TOKEN)]
                tgt_backward = [self.vocab.get(CLS_TOKEN)] + src[:-1]
                seg = [1] * len(src)
                while len(src) != self.seq_length:
                    src.append(PAD_ID)
                    tgt_forward.append(PAD_ID)
                    tgt_backward.append(PAD_ID)
                    seg.append(PAD_ID)
                pickle.dump((src, tgt_forward, tgt_backward, seg), dataset_writer)

                if pos >= end - 1:
                    break

        dataset_writer.close()


class BilmDataLoader(DataLoader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            tgt_forward = []
            tgt_backward = []
            seg = []

            for ins in instances:
                src.append(ins[0])
                tgt_forward.append(ins[1])
                tgt_backward.append(ins[2])
                seg.append(ins[3])

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt_forward), \
                torch.LongTensor(tgt_backward), \
                torch.LongTensor(seg)


class Seq2seqDataset(Dataset):
    def __init__(self, args, vocab, tokenizer):
        super(Seq2seqDataset, self).__init__(args, vocab, tokenizer)
        self.tgt_seq_length = args.tgt_seq_length
        self.src_vocab, self.src_tokenizer = vocab, tokenizer
        self.tgt_tokenizer = args.tgt_tokenizer
        self.tgt_vocab = self.tgt_tokenizer.vocab

    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        dataset_writer = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        pos = 0
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                f.readline()
                pos += 1
            while True:
                line = f.readline()
                pos += 1

                if len(line.strip().split("\t")) != 2:
                    if pos >= end:
                        break
                    continue
                document_src, document_tgt = line.strip().split("\t")
                src = self.src_tokenizer.convert_tokens_to_ids(self.src_tokenizer.tokenize(document_src))
                tgt = self.tgt_tokenizer.convert_tokens_to_ids(self.tgt_tokenizer.tokenize(document_tgt))

                src = [self.src_vocab.get(CLS_TOKEN)] + src + [self.src_vocab.get(SEP_TOKEN)]
                tgt = [self.tgt_vocab.get(CLS_TOKEN)] + tgt + [self.tgt_vocab.get(SEP_TOKEN)]
                seg = [1] * len(src)

                src, tgt, seg = src[:self.seq_length], tgt[:self.tgt_seq_length + 1], seg[:self.seq_length]
                while len(src) != self.seq_length:
                    src.append(PAD_ID)
                    seg.append(PAD_ID)
                while len(tgt) != self.tgt_seq_length + 1:
                    tgt.append(PAD_ID)
                pickle.dump((src, tgt, seg), dataset_writer)

                if pos >= end:
                    break

            dataset_writer.close()


class Seq2seqDataLoader(DataLoader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            tgt_in = []
            tgt_out = []
            seg = []

            for ins in instances:
                src.append(ins[0])
                tgt_in.append(ins[1][:-1])
                tgt_out.append(ins[1][1:])
                seg.append(ins[2])

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt_in), \
                torch.LongTensor(tgt_out), \
                torch.LongTensor(seg)


class T5Dataset(MlmDataset):
    '''
    T5 can reuse the code of MlmDataset.
    '''
    pass


class T5DataLoader(DataLoader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            tgt_in = []
            tgt_out = []
            seg = []

            tgt_seq_length = 0

            for _, ins in enumerate(instances):
                if len(ins) == 3:
                    src_single = ins[0]
                    tgt_single = ins[1]
                    seg.append([1] * ins[2][0] + [PAD_ID] * (len(ins[0]) - ins[2][0]))
                else:
                    src_single, tgt_single = mask_seq(ins[0], self.tokenizer, self.whole_word_masking, self.span_masking, self.span_geo_prob, self.span_max_length)
                    seg.append([1] * ins[1][0] + [PAD_ID] * (len(ins[0]) - ins[1][0]))

                MASK_ID = self.vocab.get(MASK_TOKEN)
                SENTINEL_ID = self.vocab.get(SENTINEL_TOKEN)

                for src_index, _ in tgt_single:
                    if src_single[src_index] != MASK_ID:
                        src_single[src_index] = MASK_ID

                tgt_in_single = [self.vocab.get(CLS_TOKEN)]
                mask_index = 0
                src_with_sentinel = []
                for token_id in src_single:
                    if token_id == MASK_ID:
                        if len(src_with_sentinel) > 0 and src_with_sentinel[-1] == (SENTINEL_ID - 1):
                            pass
                        else:
                            src_with_sentinel.append(SENTINEL_ID)
                            tgt_in_single.append(SENTINEL_ID)
                            SENTINEL_ID += 1
                        tgt_in_single.append(tgt_single[mask_index][1])
                        mask_index += 1
                    else:
                        src_with_sentinel.append(token_id)
                tgt_in_single.append(SENTINEL_ID)
                tgt_in_single.append(self.vocab.get(SEP_TOKEN))

                while len(src_with_sentinel) < len(src_single):
                    src_with_sentinel.append(PAD_ID)

                if len(tgt_in_single) > tgt_seq_length:
                    tgt_seq_length = len(tgt_in_single)

                src.append(src_with_sentinel)
                tgt_in.append(tgt_in_single)
                tgt_out.append(tgt_in[-1][1:] + [PAD_ID])

            for i in range(len(tgt_in)):
                while len(tgt_in[i]) != tgt_seq_length:
                    tgt_in[i].append(PAD_ID)
                    tgt_out[i].append(PAD_ID)

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt_in), \
                torch.LongTensor(tgt_out), \
                torch.LongTensor(seg)


class ClsDataset(Dataset):
    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        f_write = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        pos = 0
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                line = f.readline()
                pos += 1
            while True:
                line = f.readline()
                pos += 1

                line = line.strip().split('\t')
                if len(line) == 2:
                    label = int(line[0])
                    text = " ".join(line[1:])
                    src = [self.vocab.get(t) for t in self.tokenizer.tokenize(text)]
                    src = [self.vocab.get(CLS_TOKEN)] + src
                    tgt = label
                    seg = [1] * len(src)
                    if len(src) >= self.seq_length:
                        src = src[:self.seq_length]
                        seg = seg[:self.seq_length]
                    else:
                        while len(src) != self.seq_length:
                            src.append(PAD_ID)
                            seg.append(PAD_ID)
                    pickle.dump((src, tgt, seg), f_write)
                elif len(line) == 3:  # For sentence pair input.
                    label = int(line[0])
                    text_a, text_b = line[1], line[2]

                    src_a = [self.vocab.get(t) for t in self.tokenizer.tokenize(text_a)]
                    src_a = [self.vocab.get(CLS_TOKEN)] + src_a + [self.vocab.get(SEP_TOKEN)]
                    src_b = [self.vocab.get(t) for t in self.tokenizer.tokenize(text_b)]
                    src_b = src_b + [self.vocab.get(SEP_TOKEN)]

                    src = src_a + src_b
                    seg = [1] * len(src_a) + [2] * len(src_b)

                    if len(src) >= self.seq_length:
                        src = src[:self.seq_length]
                        seg = seg[:self.seq_length]
                    else:
                        while len(src) != self.seq_length:
                            src.append(PAD_ID)
                            seg.append(PAD_ID)
                    pickle.dump((src, tgt, seg), f_write)
                else:
                    pass

                if pos >= end - 1:
                    break

        f_write.close()


class ClsDataLoader(DataLoader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            tgt = []
            seg = []

            for ins in instances:
                src.append(ins[0])
                tgt.append(ins[1])
                seg.append(ins[2])

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt), \
                torch.LongTensor(seg)


class PrefixlmDataset(Dataset):

    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        dataset_writer = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        pos = 0
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                f.readline()
                pos += 1
            while True:
                line = f.readline()
                pos += 1

                if len(line.strip().split("\t")) != 2:
                    if pos >= end:
                        break
                    continue
                document_src, document_tgt = line.strip().split("\t")
                src = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(document_src))
                tgt = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(document_tgt))
                src = [self.vocab.get(CLS_TOKEN)] + src + [self.vocab.get(SEP_TOKEN)]
                tgt = tgt + [self.vocab.get(SEP_TOKEN)]
                seg_pos = [len(src)]

                if seg_pos[0] >= self.seq_length:
                    continue

                src = src + tgt
                tgt = [0] * seg_pos[0] + tgt[1:] + [PAD_ID]
                seg_pos.append(len(src))
                src, tgt = src[:self.seq_length], tgt[:self.seq_length]
                while len(src) != self.seq_length:
                    src.append(PAD_ID)
                    tgt.append(PAD_ID)
                if seg_pos[1] > self.seq_length:
                    seg_pos[1] = self.seq_length

                pickle.dump((src, tgt, seg_pos), dataset_writer)

                if pos >= end:
                    break

            dataset_writer.close()


class PrefixlmDataLoader(DataLoader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            tgt = []
            seg = []

            for ins in instances:
                src.append(ins[0])
                tgt.append(ins[1])
                seg.append([1] * ins[2][0] + [2] * (ins[2][1] - ins[2][0]) + [PAD_ID] * (len(ins[0]) - ins[2][1]))

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt), \
                torch.LongTensor(seg)

