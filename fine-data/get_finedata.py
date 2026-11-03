import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
sys.path.append(os.getcwd())
from finetuning_data_gen import (
    generation_multiP, 
)
def main(dataset):
    # 设置路径
    pcap_path = os.path.expanduser(f"~/llmfortraffic/dataPcap_flow_/{dataset}")          # pcap文件路径
    
    # # 1. 分解流量
    # print("Step 1: Converting and splitting pcap files...")
    # convert_splitcap(
    #     pcapng_path=pcapng_path,
    #     is_pcap_label=True
    # )
    '''
    malware 477
    benign 3577
    all flows:  4054 2
    '''


    # 2. 生成JSON文件
    max_packets = 20    #每个流最多保留的包数量
    enhance_factor = 1
    samples = 1000     #每个标签最多保留的样本数。
    min_flow = 1 #最小的流含包数量
    traceMaxFlow = 1000 #每个踪迹保留1000个流
    print("Step 2: get packet data json...")
    generation_multiP(
        dataset_path=pcap_path,
        samples =samples,
        enhance_factor=enhance_factor,
        max_packets=max_packets,
        min_flow=min_flow,
        traceMaxFlow=traceMaxFlow
    )                             


if __name__ == "__main__":

    dataset_list = ["cipherspectrum_aes128", "cipherspectrum_aes256", 
                 "cipherspectrum_cha", "cipherspectrum_mix", "cstent", "Edge-IIoTset",'ustc']
    dataset_list = ["itcnet"]
    for dataset in dataset_list:
        print(f"----------{dataset}----------")
        main(dataset)
    '''
    1. pcapng_path = "../../dataPcap/isd_5s"          # 原始pcapng文件路径
    2. max_packets = 20    #每个流最多保留的包数量
    3. min_flow = 5 #每个流最少含包的数量
    '''