import random
from copy import deepcopy

class FieldNegativeGenerator:
    def __init__(self):
        # 字段顺序定义（IP头→TCP头/UDP头）
        self.ip_fields = [
            "time_delta",  # 与前一包的时间差值（毫秒）
            "direction", 
            # "version", 
            # "headlen", 
            "tos", 
            "len",
            # "id", 
            "flags", 
            # "frag", 
            "ttl", 
            "proto"
        ]
        self.tcp_fields = [
            # "sport", 
            # "dport", 
            # "seq_low", 
            # "ack_low", 
            "dataofs", 
            # "res",
            "flags", 
            "window"
        ]
        self.udp_fields = [
            # "sport", 
            # "dport", 
            "len", 
            "reserved1", 
            "reserved2", 
            # "reserved3",
            # "reserved4", 
            # "reserved5"
        ]
        self.total_fields = len(self.ip_fields) + max(len(self.tcp_fields), len(self.udp_fields))
        # 核心约束定义（基于协议规范）
        self.constraints = {
            # IP头字段约束
            "ip": {
                # 时间差值：非负（≥0），通常不超过5000ms（可根据数据调整）
                # "time_delta": {"type": "range", "min": "0000", "max": "1388"},  # 0-5000ms（0x1388=5000）
                # "direction": {"type": "enum", "values": ["0001", "0002"]},  # 方向枚举
                "direction": {"type": "direction"},  # 方向枚举
                # "version": {"type": "enum", "values": ["0004"]},  # IPv4/IPv6
                # "headlen": {"type": "range", "min": "0005", "max": "000f"},  # 5-15（32位字）   20-60字节
                "tos": {"type": "range", "min": "0000", "max": "00ff"},  # DSCP(6位)+ECN(2位)
                "len": {"type": "range", "min": "0014", "max": "ffff"},  # 总长度   20-65535
                # "id": {"type": "increasing"},  # 标识递增
                # IP标志位：3位（bit0=MF, bit1=DF, bit2=保留），有效组合为以下4种
                "flags": {"type": "enum", "values": ["0000", "0001", "0002", "0003"]},
                # "frag": {"type": "range", "min": "0000", "max": "1fff"},  # 片偏移（13位）
                "ttl": {"type": "range", "min": "0001", "max": "00ff"},  # 1-255
                "proto": {"type": "enum", "values": ["0006", "0011"]}  # TCP=6, UDP=17
            },
            # TCP头字段约束（flags重点修正）
            "tcp": {
                # 端口：同一流中固定，无效值=与历史端口不同
                # "sport": {"type": "fixed_in_flow"},
                # "dport": {"type": "fixed_in_flow"},
                # "seq_low": {"type": "increasing"},  # 序列号递增
                # "ack_low": {"type": "increasing"},  # 确认号递增
                "dataofs": {"type": "range", "min": "0005", "max": "000f"},  # 5-15  20-60字节
                # "res": {"type": "fixed", "value": "0000"},  # 保留位必须为0
                # TCP标志位：9位（ns,cwr,ece,urg,ack,psh,rst,syn,fin），有效组合如下
                "flags": {"type": "enum", "values": [
                    # "0000",  # 无标志     合法，但罕见；只能出现在完全空的“伪”报文（如某些探测或教学用 raw socket）。正常协议栈永远不会发出。
                    # "0001",  # FIN        合法但罕见；正常关闭必须带 ACK（FIN+ACK）。裸 FIN 会被栈丢弃。
                    "0002",  # SYN 1    合法，非常常见：主动打开一方发送纯 SYN开始三次握手。
                    "0004",  # RST  1   合法且常见：端口未监听、连接异常、防火墙拒绝服务时都发裸 RST。
                    # "0008",  # PSH    合法但无意义；没有 ACK 的纯 PSH 会被任何栈直接丢弃。属于“畸形/测试”流量。
                    "0010",  # ACK 1    合法 + 海量存在：所有携带数据或确认的普通报文（除第一次握手外）都只有 ACK。
                    # "0020",  # URG  合法但功能已废弃；现代 TCP 实现忽略紧急指针，真实流量中几乎看不到。
                    # "0040",  # ECE  合法但不会单独出现。ECE 需要前期 ECN 握手（SYN 里置 ECE/CWR），裸 ECE 会被丢弃。
                    # "0080",  # CWR  合法但极少单独出现。CWR 通常和 ECE 成对使用，单独发没有意义，主流栈不会生成。
                    # "0100",  # NS   合法（RFC 3540），但几乎见不到；需要两端都开启 ECN-Nonce 实验特性，现实网络 0-day 级别罕见。
                    "0012",  # SYN+ACK  1
                    "0011",  # FIN+ACK  1
                    "0018",  # PSH+ACK 1
                    "0014",  # RST+ACK 1
                ]},
                "window": {"type": "range", "min": "0001", "max": "ffff"},  # 窗口>0
                # "urgptr": {"type": "range", "min": "0000", "max": "ffff"}
            },
            # UDP头字段约束（端口固定性）
            "udp": {
                # 端口：同一流中固定，无效值=与历史端口不同
                # "sport": {"type": "fixed_in_flow"},
                # "dport": {"type": "fixed_in_flow"},
                "len": {"type": "range", "min": "0008", "max": "ffeb"},  # 至少8字节（UDP头长度），减去 IPv4 头 20 字节 → UDP 报文最大 65515 字节
                "reserved1": {"type": "fixed", "value": "0000"},  # 保留字段必须为0
                "reserved2": {"type": "fixed", "value": "0000"},
                # "reserved3": {"type": "fixed", "value": "0000"},
                # "reserved4": {"type": "fixed", "value": "0000"},
                # "reserved5": {"type": "fixed", "value": "0000"},
            }
        }
        # self.flow_context = {"id_out":None, "id_in":None}  # 流上下文（含历史端口信息）

    def generate_negative_sample(self, modified, protocol_type,  modify_count, field_mask_list, packet_idex):
        """
        生成负样本并标记修改位置
        :param original_fields: 原始字段列表（ip字段+tcp/udp字段）
        :param protocol_type: 协议类型（"tcp"/"udp"）
        :param flow_context: 流上下文（含历史端口信息），格式：
            {
                "flow_id": "xxx",
                "history_sports": ["dd02", ...],  # 该流历史源端口
                "history_dports": ["01bb", ...]   # 该流历史目的端口
            }
        :param modify_count: 修改字段数量
        :return: (修改后字段列表, 修改标记列表)
        """
        protocol_type = protocol_type.lower()
        if protocol_type.lower() not in ["tcp", "udp"]:
            raise ValueError("协议类型必须为'tcp'或'udp'")
        if len(modified) != self.total_fields:
            raise ValueError(f"字段数量错误，应为{self.total_fields}个")
        # modified = deepcopy(original_fields)
        # mask = [0] * self.total_fields  # 0=未改，1=已改
        proto_fields = self.tcp_fields if protocol_type.lower() == "tcp" else self.udp_fields

        # 随机选择修改字段索引
        all_indices = list(range(self.total_fields))
        selected_indices = random.sample(all_indices, min(modify_count, len(all_indices)))
        # driection = modified[self.ip_fields.index("direction")]
        for idx in selected_indices:
            # 确定字段类型和名称
            if idx < len(self.ip_fields):
                field_type = "ip"
                field_name = self.ip_fields[idx]
            else:
                field_type = protocol_type
                field_name = proto_fields[idx - len(self.ip_fields)]
            
            # 生成无效值
            current_val = modified[idx] #字段原始值
            if field_name not in self.constraints[field_type]:
                continue  # 无约束，跳过
            if field_name == "window" and current_val == 5:
                continue # 窗口为0时不改
            constraints = self.constraints[field_type][field_name]  #有效值范围

            # if current_val == 0:
                # pass   # 窗口为0时不改
            invalid_val = self._get_invalid_value(current_val, field_name, constraints)
            src_index = idx + 1 + packet_idex*(self.total_fields+1)
            field_mask_list.append((src_index, modified[idx]))
            modified[idx] = invalid_val     #非法值


    def _get_invalid_value(self, current_val, field_name, constraints):
        """根据字段约束生成无效值
            current_val: 当前字段值（十六进制字符串）
            field_name: 字段名称（如"sport"）
            constraints: 该字段的约束字典
        """
        current_val = current_val - 5 # token id 转为原始值
        # 辅助转换函数
        hex_to_int = lambda h: int(h, 16)
        int_add_five = lambda i: i + 5  # 原始值转为 token id

        # 处理端口（固定性约束）
        # if field_name in ["sport", "dport"] and constraints["type"] == "fixed_in_flow":
        #     while True:
        #         invalid = (random.randint(0, 65535))
        #         if invalid != current_val:
        #             return int_add_five(invalid)

        if constraints["type"] == "enum":
            while True:
                invalid = (random.randint(0, 0xffff))
                if invalid not in constraints["values"] and invalid != current_val:
                    return int_add_five(invalid)
        elif constraints["type"] == "range":
            min_val = hex_to_int(constraints["min"])
            max_val = hex_to_int(constraints["max"])
            while True:
                invalid_int = random.choice([
                    random.randint(0, min_val - 1) if min_val > 0 else None,        #一个小于下限的或者None
                    random.randint(max_val + 1, 0xffff) if max_val < 0xffff else None   #一个大于上限的或者None
                ])
                if invalid_int is not None:
                    invalid = (invalid_int)
                    if invalid != current_val:
                        return int_add_five(invalid)
        elif constraints["type"] == "fixed":
            invalid = (random.randint(0, 0xffff))
            while True:
                invalid = (random.randint(0, 0xffff))
                if invalid != current_val:
                    return int_add_five(invalid)
        # elif constraints["type"] == "increasing":
        #     if current_val == 0:
        #         return int_add_five(random.randint(1, 0xffff))  # 破坏递增（后续包更大）
        #     else:
        #         return int_add_five(random.randint(0, current_val - 1))  # 小于当前值
        elif constraints["type"] == "direction":
            return int_add_five(2 if current_val == 1 else 1)