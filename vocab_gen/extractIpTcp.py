import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys
from scapy.all import rdpcap, IP, TCP, UDP, PacketList  # 新增UDP导入
# import json
import tqdm
import multiprocessing as mp


class DirectionBasedHeaderProcessor:
    def __init__(self):
        # 所有字段标准化为16位(4个十六进制字符)
        self.standard_length = 4  # 4个十六进制字符 = 16位
        self.timestamp_standard_length = 4  # 时间戳标准化为8个十六进制字符 = 32位
        self.timestamp_max = 0xFFFF  # 时间戳最大值（32位）
        # --------------------------
        # 1. 公共字段：IP头（TCP/UDP共用）
        # --------------------------
        self.ip_fields = [
            #时序模式是流量分类的灵魂。
            # 不同应用（如视频流、文件传输、交互式网页）的包到达间隔（IAT）分布差异显著。
            # 例如：视频流呈周期性突发，网页浏览呈间歇性。
            "timestamp",    # 时间差        *

            #上下行不对称性是区分应用的关键。
            # 如直播上行小、下行大；P2P 上下行对称；远程 Shell 上行大、下行小。
            "direction",     # 方向标识 → 16位  *


            #现代网络几乎全是 IPv4（值为 4），无区分度。
            # "version",  # 版本(4位)+ → 16位         无意义

            #标准 IP 头为 20 字节（值=5），极少变化（除非有 IP 选项，但现代流量中罕见）。
            # "headlen",      # 首部长度(16位)        无意义

            #若网络中启用了 QoS 策略，可反映业务优先级（如语音流标记为 EF），
            # 但多数网络中该字段被忽略或统一置 0。  现多用作 DSCP 进行 QoS 标记
            "tos",          # 服务类型(8位) → 扩展为16位        **

            #包大小分布（Packet Size Distribution）是经典流量特征。
            # MTU 探测、分片行为、应用层协议单元大小均可体现。
            "len",          # 总长度(16位)      *

            #高度动态噪声。由操作系统内核每发包递增（或随机化），
            #用于分片重组，不携带应用语义。将其输入模型等于输入一个递增序列，无分类价值。
            # "id",           # 标识(16位)    无意义

            #DF（Don't Fragment）位常用于路径 MTU 发现。
            # 某些协议（如 QUIC）会设置 DF，但总体区分度有限。
            "flags",        # 标志(3位)->16位       **

            #现代网络中分片极少见（因 PMTU 发现和 DF 位普及），绝大多数为 0。
            #现代网络的思路是：不要让路由器分片，让发送端主机自己控制包大小。
            #现代流量中 frag 几乎总是 0
            # "frag",         # 片偏移(13位)      无意义

            #可粗略推断目标 OS 类型（如 Windows 默认 128，Linux 默认 64）和路径跳数，
            # 但会随网络拓扑动态变化，稳定性一般。
            "ttl",          # 生存时间(8位) → 扩展为16位        **

            #区分 TCP/UDP/ICMP 等基础传输模式，是首要分类维度。
            "proto",        # 协议(8位) → 扩展为16位（6=TCP，17=UDP）   *
            # "chksum"       # 校验和(16位) 校验和同样依赖伪首部和 payload，你的场景中无法计算有效校验值，与其他字段无真实关联，无需学习。
        ]
        
        # --------------------------
        # 2. TCP头字段（原有逻辑保留）
        # --------------------------
        self.tcp_fields = [

            #在深度学习训练流程中，端口号已经被完全随机化
            # "sport",        # 源端口(16位)      无意义
            # "dport",        # 目的端口(16位)    无意义

            #强噪声。序列号和确认号由内核根据初始序列号（ISN，通常随机化）生成，
            #呈现伪随机特性，与流量类型无关。模型可能过拟合这些无意义数值。
            # "seq_low",      # 序列号低16位          无意义
            # "ack_low",      # 确认号低16位          无意义  

            #反映 TCP 选项使用情况（如时间戳、SACK、窗口缩放）。
            #不同 OS 的 TCP 选项组合不同，可作为操作系统指纹的一部分，但信息量较小。
            "dataofs",      # 数据偏移(4位)+        **

            #保留位，协议规定必须为 0，无信息
            # "res",          # 保留位(6位)         无意义

            #状态机指纹。
            # 例如：扫描行为（SYN→RST）、正常握手（SYN→SYN-ACK→ACK）、连接终止（FIN/ACK）、心跳保活。
            # 不同应用有不同的标志位组合模式。
            "flags",        # 标志(6位) → 扩展为16位    *

            #操作系统/应用指纹。
            #不同 OS 的初始窗口大小不同（如 Windows  vs Linux），
            #且窗口缩放行为反映应用数据产生速率。
            "window",       # 窗口大小(16位)    *



            # "urgptr"        # 紧急指针(16位)  极少使用，可以不考虑
            # "chksum",       # 校验和(16位)    校验和同样依赖伪首部和 payload，你的场景中无法计算有效校验值，与其他字段无真实关联，无需学习。
        ]
        
        # --------------------------
        # 3. 新增：UDP头字段（与TCP字段数量对齐，便于后续处理）
        # --------------------------
        self.udp_fields = [
            #在深度学习训练流程中，端口号已经被完全随机化
            # "sport",        # 源端口(16位) → 与TCP一致      无意义
            # "dport",        # 目的端口(16位) → 与TCP一致    无意义


            #UDP 无流量控制，其长度直接反映应用层数据单元大小
            #（如 DNS 查询通常 <512 字节，QUIC 数据包通常较大）。
            "len",          # UDP总长度(16位)   *
            "reserved1",    # 保留字段1(16位) → 填充，与TCP对齐（UDP无该字段）
            "reserved2",    # 保留字段2(16位) → 填充
            # "reserved3",    # 保留字段3(16位) → 填充）
            # "reserved4",    # 保留字段4(16位) → 填充
            # "reserved5",     # 保留字段5(16位) → 填充
            # "reserved6"     # 保留字段6(16位) → 填充
            # "chksum",       # UDP校验和(16位)  校验和同样依赖伪首部和 payload，你的场景中无法计算有效校验值，与其他字段无真实关联，无需学习。
        ]
        
        # 存储当前流的所有包头数据（1个PCAP文件=1个流，含TCP/UDP）
        self.extracted_headers = []
        self.prev_timestamp = None # 上一个包的时间戳（用于计算时间差）
        
        # 流的基准方向（第一个包的IP+端口+协议，区分流）
        self.base_src_ip = None
        self.base_dst_ip = None
        self.base_sport = None
        self.base_dport = None
        self.base_proto = None  # 新增：基准协议（TCP/UDP），避免跨协议误判
        
        # 方向标识的十六进制映射（原有逻辑保留）
        self.dir_hex_map = {
            "OUT": "0001",  # 发包
            "IN": "0002"    # 收包
        }
        
        # 协议号映射（IP头proto字段值）
        self.proto_map = {
            6: "TCP",   # TCP协议号=6
            17: "UDP"   # UDP协议号=17
        }
    # --------------------------
    # 时间戳标准化（32位/8个十六进制字符，毫秒级差值处理）
    # --------------------------
    def _standardize_timestamp(self, timestamp_diff_ms):
        if timestamp_diff_ms < 0:   #包乱序
            timestamp_diff_ms = 0
        elif timestamp_diff_ms > self.timestamp_max:
            timestamp_diff_ms = self.timestamp_max
        
        hex_str = f"{int(timestamp_diff_ms):x}"     #  :x 表示将整数转换为小写十六进制（hexadecimal）字符串。
        if len(hex_str) < self.timestamp_standard_length:
            return hex_str.zfill(self.timestamp_standard_length)
        return hex_str[-self.timestamp_standard_length:]  # 保留低位
    # --------------------------
    # 计算当前包的时间戳差值（毫秒级）
    # --------------------------
    def _calc_timestamp_diff(self, current_packet_time):
        current_ts_ms = int(round(current_packet_time * 1000))  # 当前包时间戳（毫秒）
        if self.prev_timestamp is None:
            self.prev_timestamp = current_ts_ms
            return 0
        
        diff_ms = current_ts_ms - self.prev_timestamp
        self.prev_timestamp = current_ts_ms
        return diff_ms
    # --------------------------
    # 4. 修改：方向判断（增加协议匹配，避免TCP/UDP跨协议误判）
    # --------------------------
    def _determine_direction(self, ip_layer, transport_layer, proto):
        """
        以当前流的第一个包为基准，判断后续包的方向（增加协议校验）
        proto: 传输层协议（"TCP"或"UDP"）
        """
        # 第一个包：初始化流的基准信息（含协议）
        if self.base_src_ip is None:
            self.base_src_ip = ip_layer.src
            self.base_dst_ip = ip_layer.dst
            self.base_sport = transport_layer.sport
            self.base_dport = transport_layer.dport
            self.base_proto = proto  # 记录基准协议
            return "OUT"
        
        # 后续包：必须与基准协议一致，否则视为未知方向
        if proto != self.base_proto:
            return "UNKNOWN"
        
        # 协议一致时，匹配IP+端口方向
        if (ip_layer.src == self.base_src_ip and 
            ip_layer.dst == self.base_dst_ip and 
            transport_layer.sport == self.base_sport and 
            transport_layer.dport == self.base_dport):
            return "OUT"
        elif (ip_layer.src == self.base_dst_ip and 
              ip_layer.dst == self.base_src_ip and 
              transport_layer.sport == self.base_dport and 
              transport_layer.dport == self.base_sport):
            return "IN"
        else:
            return "UNKNOWN"  # 无法匹配当前流的包


    # --------------------------
    # 5. 原有逻辑保留：字段标准化
    # --------------------------
    def _standardize_field(self, value, is_numeric=True):
        if is_numeric:
            hex_str = f"{value:x}"  # 数值转十六进制
        else:
            hex_str = self.dir_hex_map.get(value, "0000") if value != "UNKNOWN" else "0000"
        
        # 长度校准（超长截断、不足补0）
        if len(hex_str) > self.standard_length:
            return hex_str[-self.standard_length:]  # 保留低位
        return hex_str.zfill(self.standard_length)  # 前补0至4位十六进制，就是16位二进制


    # --------------------------
    # 6. 原有逻辑保留：IP头处理（增加协议字段传递）
    # --------------------------
    def _process_ip_header(self, ip_layer, transport_layer, proto, timestamp_diff_ms):
        ip_data = {}
        # 版本+首部长度（合并为16位）
        # ip_data["version"] = self._standardize_field(ip_layer.version)
        # ip_data["headlen"] = self._standardize_field(ip_layer.ihl)
        # 服务类型（8位→16位）
        ip_data["tos"] = self._standardize_field(ip_layer.tos)
        # 总长度、标识、标志+片偏移
        ip_data["len"] = self._standardize_field(ip_layer.len)
        # ip_data["id"] = self._standardize_field(ip_layer.id)
        ip_data["flags"] = self._standardize_field(ip_layer.flags.value)
        # ip_data["frag"] = self._standardize_field(ip_layer.frag)
        # TTL、协议（6=TCP，17=UDP）、校验和
        ip_data["ttl"] = self._standardize_field(ip_layer.ttl)
        ip_data["proto"] = self._standardize_field(ip_layer.proto)  # 协议号标准化
        # ip_data["chksum"] = self._standardize_field(ip_layer.chksum)
        # 方向标识（传入协议，避免跨协议误判）
        direction = self._determine_direction(ip_layer, transport_layer, proto)
        ip_data["direction"] = self._standardize_field(direction, is_numeric=False)
        # 时间戳差值（毫秒级）
        ip_data["timestamp"] = self._standardize_timestamp(timestamp_diff_ms)
        return ip_data


    # --------------------------
    # 7. 原有逻辑保留：TCP头处理
    # --------------------------
    def _process_tcp_header(self, tcp_layer):
        tcp_data = {}
        # 源端口、目的端口
        # tcp_data["sport"] = self._standardize_field(tcp_layer.sport)
        # tcp_data["dport"] = self._standardize_field(tcp_layer.dport)
        # 序列号低16位、确认号低16位
        # tcp_data["seq_low"] = self._standardize_field(tcp_layer.seq & 0xFFFF)
        # tcp_data["ack_low"] = self._standardize_field(tcp_layer.ack & 0xFFFF)
        # 数据偏移+保留位（4+6=10位→16位）
        tcp_data["dataofs"] = self._standardize_field(tcp_layer.dataofs)
        # tcp_data["res"] = self._standardize_field(tcp_layer.reserved)
        # 标志、窗口大小、校验和、紧急指针
        tcp_data["flags"] = self._standardize_field(tcp_layer.flags.value)
        tcp_data["window"] = self._standardize_field(tcp_layer.window)
        # tcp_data["chksum"] = self._standardize_field(tcp_layer.chksum)
        # tcp_data["urgptr"] = self._standardize_field(tcp_layer.urgptr)
        return tcp_data


    # --------------------------
    # 8. 新增：UDP头处理（与TCP字段数量对齐，填充保留字段）
    # --------------------------
    def _process_udp_header(self, udp_layer):
        udp_data = {}
        # 源端口、目的端口（与TCP字段名一致）
        # udp_data["sport"] = self._standardize_field(udp_layer.sport)
        # udp_data["dport"] = self._standardize_field(udp_layer.dport)
        # UDP总长度、UDP校验和（替代TCP的序列号、确认号）
        udp_data["len"] = self._standardize_field(udp_layer.len)
        # udp_data["chksum"] = self._standardize_field(udp_layer.chksum if udp_layer.chksum else 0)  # 处理无校验和的情况
        # 保留字段（填充0，与TCP字段数量对齐，确保后续拼接长度一致）
        udp_data["reserved1"] = self._standardize_field(0)
        udp_data["reserved2"] = self._standardize_field(0)
        # udp_data["reserved3"] = self._standardize_field(0)
        # udp_data["reserved4"] = self._standardize_field(0)
        # udp_data["reserved5"] = self._standardize_field(0)
        # udp_data["reserved6"] = self._standardize_field(0)
        return udp_data


    # --------------------------
    # 9. 修改：生成分组字符串（支持TCP/UDP两种协议）
    # --------------------------
    def _generate_grouped_string(self, ip_data, transport_data, proto):
        """
        生成符合格式的字符串（支持TCP/UDP）
        proto: 传输层协议（"TCP"或"UDP"）
        """
        # 1. 按顺序拼接所有字段的十六进制值（IP在前，传输层在后）
        full_hex = ""
        # 拼接IP字段（TCP/UDP共用）
        for field in self.ip_fields:
            full_hex += ip_data[field]
        # 拼接传输层字段（根据协议选择TCP/UDP字段）
        if proto == "TCP":
            for field in self.tcp_fields:
                full_hex += transport_data[field]
        elif proto == "UDP":
            for field in self.udp_fields:
                full_hex += transport_data[field]
        
        # 2. 每4个字符（2字节）为一组，组间加空格（与原需求一致）
        # 示例："00010203" → ["0001", "0203"] → "0001 0203"
        hex_groups = [full_hex[i:i+4] for i in range(0, len(full_hex), 4) if full_hex[i:i+4]]
        return " ".join(hex_groups)


    # --------------------------
    # 10. 修改：包头提取（支持TCP/UDP筛选，增加协议判断）
    # --------------------------
    def _read_pcap(self, pcap_input):
        if isinstance(pcap_input, str):
            return rdpcap(pcap_input)
        elif isinstance(pcap_input, list) or isinstance(pcap_input, PacketList):
            return pcap_input
        else:
            return list([pcap_input])

    def extract_headers_from_pcap(self, pcap_file, max_packets=None):
        try:
            packets = self._read_pcap(pcap_file)
            count = 0
            total_packets = 0
            filtered_packets = 0  # 非IP/TCP/UDP的包计数
            self.__reset_stream()  # 重置流状态
            
            for packet in packets:
                if max_packets and count >= max_packets:
                    break
                total_packets += 1
                try:
                    # 筛选：仅处理含IP层且含TCP/UDP层的包
                    if packet.haslayer(IP):
                        ip_layer = packet.getlayer(IP)
                        if ip_layer.version != 4:   #跳过ipv6包
                            filtered_packets += 1
                            continue
                        proto = ip_layer.proto  # 获取IP层协议号（6=TCP，17=UDP）
                        
                        # 判断传输层协议
                        if proto == 6 and packet.haslayer(TCP):
                            transport_layer = packet.getlayer(TCP)
                            transport_proto = "TCP"
                        elif proto == 17 and packet.haslayer(UDP):
                            transport_layer = packet.getlayer(UDP)
                            transport_proto = "UDP"
                        else:
                            # 非TCP/UDP的IP包（如ICMP），过滤
                            filtered_packets += 1
                            continue
                    else:
                        # 非IP包（如ARP），过滤
                        filtered_packets += 1
                        continue
                    
                    # 计算时间戳差值（毫秒级）
                    timestamp_diff_ms = self._calc_timestamp_diff(packet.time)
                    # 处理IP头和传输层头（TCP/UDP分支）
                    ip_data = self._process_ip_header(ip_layer, transport_layer, transport_proto, timestamp_diff_ms)
                    if transport_proto == "TCP":
                        transport_data = self._process_tcp_header(transport_layer)
                    else:
                        transport_data = self._process_udp_header(transport_layer)
                    
                    # 跳过方向未知的包（非当前流的包）
                    if ip_data["direction"] == "0000" and count > 0:
                        print(f"警告：{pcap_file}的第{count+1}个{transport_proto}包方向未知，已跳过")
                        continue
                    
                    # 生成分组字符串，加入当前流的包头列表
                    grouped_str = self._generate_grouped_string(ip_data, transport_data, transport_proto)
                    self.extracted_headers.append(grouped_str)
                    count += 1
                
                except Exception as e:
                    # 捕获解析错误，避免程序中断
                    print(f"解析{pcap_file}的第{total_packets}个包出错: {str(e)}，已跳过")
                    filtered_packets += 1
            
            # 打印详细统计，便于排查
            if isinstance(pcap_file, str):
                print(f"从{pcap_file}（1个流）中提取了{count}个有效IP+TCP/UDP包头")
                print(f"  总包数: {total_packets}, 过滤/错误包数: {filtered_packets}")
            # else:
            #     print(f"{count}个有效IP+TCP/UDP包头")
            return count
        
        except Exception as e:
            print(f"处理{pcap_file}出错: {str(e)}")
            return 0


    # --------------------------
    # 11. 修改：重置流状态（增加协议重置）
    # --------------------------
    def __reset_stream(self):
        self.extracted_headers = []
        self.base_src_ip = None
        self.base_dst_ip = None
        self.base_sport = None
        self.base_dport = None
        self.base_proto = None  # 重置基准协议
        self.prev_timestamp = None  # 重置时间戳记录，避免跨流干扰


    # --------------------------
    # 12. 原有逻辑保留：语料生成（无需修改，兼容TCP/UDP）
    # --------------------------
    def generate_bpe_corpus(self, output_file, is_multi=False):
        if is_multi:
            pid = os.getpid()
            file_path = f"{output_file}_{pid}.txt"
        else:
            file_path = output_file
        
        with open(file_path, "a", encoding="utf-8") as f:
            is_first_packet = True  # 标记当前流的第一个包
            for packet_str in self.extracted_headers:
                if is_first_packet:
                    f.write(f"\n||{packet_str}\n")
                    is_first_packet = False
                else:
                    f.write(f"{packet_str}\n")


# --------------------------
# 后续函数（extractPcap、merge_temp_files、pretrain_dataset_generation）无需修改
# --------------------------
def extractPcap(pcap_file, corpus_prefix="direction_bpe_corpus", is_multi=False):
    processor = DirectionBasedHeaderProcessor()
    processor.extract_headers_from_pcap(pcap_file, max_packets=30)
    processor.generate_bpe_corpus(corpus_prefix, is_multi=is_multi)


def merge_temp_files(corpus_prefix, temp_dir):
    final_corpus_path = f"{corpus_prefix}.txt"
    temp_files = [f for f in os.listdir(temp_dir) if f.startswith(f"{corpus_prefix}_") and f.endswith(".txt")]
    
    with open(final_corpus_path, "w", encoding="utf-8") as final_f:
        for temp_file in temp_files:
            temp_path = os.path.join(temp_dir, temp_file)
            with open(temp_path, "r", encoding="utf-8") as temp_f:
                final_f.write(temp_f.read())
            os.remove(temp_path)
    
    print(f"合并完成！最终语料库路径：{final_corpus_path}")
    print(f"合并的临时文件数量：{len(temp_files)}")


def pretrain_dataset_generation(output_split_path, corpus_prefix="direction_bpe_corpus", is_multi=True):
    # pcap_dir = os.path.join(output_split_path, "splitcap")
    pcap_dir = output_split_path
    if not os.path.exists(pcap_dir):
        raise FileNotFoundError(f"PCAP目录不存在：{pcap_dir}")
    
    all_pcap_files = []
    for root, _, files in os.walk(pcap_dir):
        for file in files:
            if file.lower().endswith((".pcap", ".pcapng")):
                all_pcap_files.append(os.path.join(root, file))
    
    if not all_pcap_files:
        print("未找到任何PCAP文件！")
        return
    
    print(f"共发现{len(all_pcap_files)}个PCAP文件（每个对应1个流）")
    
    if is_multi:
        temp_dir = f"{corpus_prefix}_temp"
        os.makedirs(temp_dir, exist_ok=True)
        corpus_temp_prefix = os.path.join(temp_dir, corpus_prefix)
        
        pbar = tqdm.tqdm(total=len(all_pcap_files), desc="多进程提取包头（TCP+UDP）")
        update_pbar = lambda *args: pbar.update()
        
        pool = mp.Pool(processes=min(mp.cpu_count(), 40))
        for pcap_file in all_pcap_files:
            pool.apply_async(extractPcap, args=(pcap_file, corpus_temp_prefix, True), callback=update_pbar)
        
        pool.close()
        pool.join()
        pbar.close()
        
        print("\n开始合并多进程临时文件...")
        merge_temp_files(corpus_prefix, temp_dir)
        os.rmdir(temp_dir)
    
    else:
        final_corpus_path = f"{corpus_prefix}.txt"
        if os.path.exists(final_corpus_path):
            os.remove(final_corpus_path)
        
        for pcap_file in tqdm.tqdm(all_pcap_files, desc="单进程提取包头（TCP+UDP）"):
            extractPcap(pcap_file, final_corpus_path, is_multi=False)
        
        print(f"单进程处理完成！最终语料库路径：{final_corpus_path}")


if __name__ == "__main__":
    # 批量处理PCAP文件（支持TCP和UDP）
    pretrain_dataset_generation(
        output_split_path=os.path.expanduser("~/llmfortraffic/dataPcap_feature/cic"),
        corpus_prefix="direction_bpe_corpus_tcp_udp",  # 语料库名增加UDP标识
        is_multi=True  # 测试时使用单进程，正式可改为True
    )