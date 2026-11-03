import csv
import random
from collections import defaultdict

class PacketSeqAugmenter:
    def __init__(self):
        # 1. 字段配置（确认每个包头的字段数量：IP层+传输层）
        self.ip_fields = [
            "timestamp",    # 时间戳
            "direction",     # 方向标识 → 16位
            "tos",          # 服务类型(8位) → 扩展为16位
            "len",          # 总长度(16位)
            # "id",           # 标识(16位)
            "flags",        # 标志(3位)+
            # "frag",         # 片偏移(13位) → 16位
            "ttl",          # 生存时间(8位) → 扩展为16位
            "proto",        # 协议(8位) → 扩展为16位（6=TCP，17=UDP）
        ]  # IP层：11个（timestamp~proto）
        self.tcp_fields = [
            # "sport",        # 源端口(16位)
            # "dport",        # 目的端口(16位)
            # "seq_low",      # 序列号低16位
            # "ack_low",      # 确认号低16位
            "dataofs",      # 数据偏移(4位)+
            # "res",          # 保留位(6位) → 扩展为16位
            "flags",        # 标志(6位) → 扩展为16位
            "window",       # 窗口大小(16位)
        ]  # TCP层：8个（sport~window）→ 但示例中是8个，按实际调整
        self.udp_fields = [
            # "sport",        # 源端口(16位) → 与TCP一致
            # "dport",        # 目的端口(16位) → 与TCP一致
            "len",          # UDP总长度(16位) 
            "reserved1",    # 保留字段1(16位) → 填充，与TCP对齐（UDP无该字段）
            "reserved2",    # 保留字段2(16位) → 填充
            # "reserved3",    # 保留字段3(16位) → 填充）
            # "reserved4",    # 保留字段4(16位) → 填充
            # "reserved5",     # 保留字段5(16位) → 填充
        ]  # UDP层：8个（sport~reserved5）
        self.ip_len = len(self.ip_fields)  # 7（固定）
        self.packet_total_fields = self.ip_len + max(len(self.tcp_fields), len(self.udp_fields))  # 7+3=10（每个包头固定10字段）
        # self.src_port = None
        # self.dst_port = None
        # self.id_add = None
        # self.seq_low = None
        # self.ack_low = None
        self.window_add = None
        # 2. 可增强字段的索引（基于IP/传输层字段位置）
        # IP层可增强字段（排除version、proto）
        self.ip_aug_idx = [
            self.ip_fields.index(f) for f in ["timestamp", "ttl"]
        ]
        # TCP层可增强字段（排除res）
        self.tcp_aug_idx = [
            self.tcp_fields.index(f) for f in ["window"]
        ]
        self.udp_aug_idx = [
            
        ]


    def hex2int(self, hex_str):
        """4字符16进制转整数（如"0030"→48）"""
        return int(hex_str, 16) if hex_str not in ["[CLS]", "[SEP]"] else hex_str

    def int2hex(self, num):
        """整数转4字符16进制（补零，如48→"0030"）"""
        return f"{num:04x}"

    def parse_seq_format(self, seq_str):
        """解析序列格式：拆分[CLS]、20个包头、20个[SEP]"""
        tokens = seq_str.strip().split()  # 按空格分割所有token
        if tokens[0] != "[CLS]":
            raise ValueError(f"序列必须以[CLS]开头，当前开头：{tokens[0]}")
        total_tokens = len(tokens)
        packet_sep_total = self.packet_total_fields + 1 # 每个包头+后续[SEP]的总字段数
        packet_num = total_tokens // packet_sep_total  # 理论包头数（含[SEP]）
        # 提取20个包头（每个包头19字段）和20个[SEP]
        packets = []
        seps = []
        ptr = 1  # 跳过[CLS]，从第一个字段开始
        for _ in range(packet_num):  # 固定20个包头
            # 提取当前包头的19个字段
            packet_fields = tokens[ptr:ptr + self.packet_total_fields]
            if len(packet_fields) != self.packet_total_fields:
                raise ValueError(f"包头字段数异常：应为{self.packet_total_fields}个，实际{len(packet_fields)}个")
            packets.append(packet_fields)
            
            # 提取包头后的[SEP]
            sep = tokens[ptr + self.packet_total_fields]
            if sep != "[SEP]":
                raise ValueError(f"包头后必须是[SEP]，当前是：{sep}")
            seps.append(sep)
            
            # 移动指针到下一个包头的起始位置
            ptr += self.packet_total_fields + 1
        
        # 验证是否解析完所有token（最后一个[SEP]后无多余字段）
        if ptr != len(tokens):
            raise ValueError(f"序列格式异常：解析后剩余{len(tokens)-ptr}个多余token")
        
        return {
            "cls": "[CLS]",
            "packets": packets,  # 20个包头，每个是19字段列表
            "seps": seps,        # 20个[SEP]
            "raw_tokens": tokens
        }

    def augment_single_packet(self, packet_fields, packet_idx):
        """增强单个包头的字段（协议合规）"""
        random.seed(7)  # 初始化随机数生成器
        augmented = packet_fields.copy()
        ip_fields = augmented[:self.ip_len]  # 前7个：IP层字段
        trans_fields = augmented[self.ip_len:]  # 后3个：传输层字段

        # 1. 增强IP层字段（只改可增强索引的字段）
        for ip_idx in self.ip_aug_idx:
            hex_val = ip_fields[ip_idx]
            int_val = self.hex2int(hex_val)
            # 按IP协议规则微调
            if self.ip_fields[ip_idx] == "timestamp":
                # 时间戳：±30
                if packet_idx == 0:
                    continue
                int_val = max(0, int_val + random.randint(-1*max(30, int_val), 30))
                int_val = min(2**16-1, int_val)  # 限制在32位范围内
            elif self.ip_fields[ip_idx] == "ttl":
                # TTL：常见值（32,64,128,255）
                if random.random() < 0.9:
                    int_val = int_val  # 保持不变
                else:
                    int_val = random.choice([45, 64, 115, 50, 21])
            # 其他字段（tos/flags/frag）按合法范围微调
            ip_fields[ip_idx] = self.int2hex(int_val)
        augmented[:self.ip_len] = ip_fields

        # 2. 增强传输层字段（先判断协议类型：proto字段在IP层第10位，索引10）
        proto_hex = ip_fields[self.ip_fields.index("proto")]
        proto_val = self.hex2int(proto_hex)
        if proto_val == 6:  # TCP协议（proto=6）
            for tcp_idx in self.tcp_aug_idx:
                if tcp_idx >= len(trans_fields):
                    continue
                hex_val = trans_fields[tcp_idx]
                int_val = self.hex2int(hex_val)
                # TCP协议规则微调
                if self.tcp_fields[tcp_idx] == "window":
                    # 窗口大小：0-65535
                    if not self.window_add:
                        self.window_add = random.randint(-1*min(100,int_val), 100)
                    int_val = int_val+self.window_add
                    int_val = random.randint(0, 100) if int_val < 0 else int_val
                    int_val = int_val%(2**16)
                    # int_val = random.randint(0, 65535)
                # 写回
                trans_fields[tcp_idx] = self.int2hex(int_val)
        elif proto_val == 17:  # UDP协议（proto=17）
            for udp_idx in self.udp_aug_idx:
                if udp_idx >= len(trans_fields):
                    continue
                hex_val = trans_fields[udp_idx]
                int_val = self.hex2int(hex_val)
                # UDP协议规则微调
                trans_fields[udp_idx] = self.int2hex(int_val)
        augmented[self.ip_len:] = trans_fields

        return augmented

    def augment_seq_sample(self, seq_str):
        """增强整个20包头序列（保留[CLS]/[SEP]结构）"""
        # 1. 解析原始格式
        parsed = self.parse_seq_format(seq_str)
        # 2. 逐个增强20个包头
        augmented_packets = []
        for idx in range(len(parsed["packets"])):
            pkt = parsed["packets"][idx]
            augmented_packets.append(self.augment_single_packet(pkt, idx))
        # augmented_packets = [self.augment_single_packet(pkt) for pkt in parsed["packets"]]
        # self.src_port = None
        # self.dst_port = None
        # self.id_add = None
        # self.seq_low = None
        # self.ack_low = None
        self.window_add = None
        # 3. 重组回原始格式：[CLS] + 包头1+[SEP] + 包头2+[SEP] + ... + 包头20+[SEP]
        cz_tokens = [parsed["cls"]]
        for pkt, sep in zip(augmented_packets, parsed["seps"]):
            cz_tokens.extend(pkt)
            cz_tokens.append(sep)
        # 验证重组后长度是否与原始一致（避免格式错误）
        if len(cz_tokens) != len(parsed["raw_tokens"]):
            raise ValueError(f"重组后格式异常：原始{len(parsed['raw_tokens'])}个token，重组后{len(cz_tokens)}个")
        # 转成字符串
        return " ".join(cz_tokens)

    def balance_seq_dataset(self, data, labels):
        """平衡类别：所有类别样本数增强到最多类的数量（增强单位：整个20包头序列）"""
        # 按类别分组
        label2seqs = defaultdict(list)
        for seq, lbl in zip(data, labels):
            label2seqs[lbl].append(seq)
        # 目标数量（最多类的样本数）
        max_count = max(len(seqs) for seqs in label2seqs.values())
        max_count = max(500, max_count)  # 限制最少500个，避免过度增强
        print(f"类别平衡目标：每个类别{max_count}个样本（当前最多类样本数）")
        
        # 补全每个类别的样本
        balanced_data = []
        balanced_labels = []
        for lbl, seqs in label2seqs.items():
            current_count = len(seqs)
            if current_count >= max_count:
                # 超过目标：随机筛选max_count个
                selected = random.sample(seqs, max_count)
            else:
                # 不足目标：原始样本 + 增强样本（轮询增强避免过拟合）
                selected = seqs.copy()
                need = max_count - current_count
                for i in range(need):
                    base_seq = seqs[i % current_count]  # 轮询原始序列
                    augmented_seq = self.augment_seq_sample(base_seq)
                    selected.append(augmented_seq)
            # 合并
            balanced_data.extend(selected)
            balanced_labels.extend([lbl] * len(selected))
        
        print(f"平衡后总样本数：{len(balanced_data)}（类别数：{len(label2seqs)}）")
        return balanced_data, balanced_labels


def write_balanced_seq_tsv(data, labels, file_dir, type, ip_fields, tcp_fields, udp_fields):
    """
    适配20包头序列格式的增强+写入TSV
    参数：
        data: 原始序列列表（每个元素是"[CLS] 0000 ... [SEP]"的字符串）
        labels: 原始标签列表（每个序列对应一个label）
        file_dir: 输出目录（如"./dataset/"）
        type: 数据集类型（train/val/test）
        ip_fields/tcp_fields/udp_fields: 字段定义（用户提供的self.xxx_fields）
    """
    # 1. 初始化增强器并平衡类别
    augmenter = PacketSeqAugmenter(ip_fields, tcp_fields, udp_fields)
    # 先验证第一个样本的格式（避免批量报错）
    try:
        _ = augmenter.parse_seq_format(data[0])
        print("✅ 第一个样本格式验证通过")
    except Exception as e:
        raise ValueError(f"❌ 样本格式错误：{str(e)}") from e
    # 平衡并增强
    balanced_data, balanced_labels = augmenter.balance_seq_dataset(data, labels)

    # 2. 写入TSV（label为字符串，避免格式问题）
    tsv_data = [["label", "text_a"]]
    for lbl, seq in zip(balanced_labels, balanced_data):
        tsv_data.append([str(lbl), seq])
    
    output_path = f"{file_dir.rstrip('/')}/{type}_balanced_dataset.tsv"
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(tsv_data)
    
    print(f"✅ 已写入平衡数据集：{output_path}")
    print(f"📊 平衡后统计：{len(balanced_labels)}个样本，{len(set(balanced_labels))}个类别")
    return output_path