import os
import json
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from finetuning_data_gen import (
    dataset_extract,
)

fine_data = "fine_data_flow"
# fine_data = "test"
enhance = 1
mincount = 200  #每个标签最少样本数
#由data.json生成tsv文件
def get_json(name):
    dataset_name = name #uav_5s
    
    dateset_root = os.path.expanduser(f"~/llmfortraffic/dataPcap_flow_/{dataset_name}") #数据集路径     #../../dataPcap/uav_5s
    dateset_path = dateset_root + "/" + dataset_name + f"_codedata_enhance{enhance}.json"   #原始json文件 '../../dataPcap/uav_5s/uav_5s_data.json'
    # dateset_path = dateset_root + "/" + dataset_name + f"_codedata.json"   #原始json文件 '../../dataPcap/uav_5s/uav_5s_data.json'
    dataset_save_path = f"./{fine_data}/" + dataset_name + f"_enhance{enhance}"     # 数据集保存路径  './fine_data/uav_5s'
    dataset_save_file = dataset_save_path + "/data.json"    #  './fine_data/uav_5s/data.json'

    if not os.path.exists(dataset_save_path):
        os.makedirs(dataset_save_path)
    fine_json = dict()
    with open(dateset_path,'r') as json_file:
        content = json.load(json_file)
        label_to_idx = dict()
        for i, label in enumerate(content.keys()):  # 0 malware
            label_to_idx[label] = i
            fine_json[str(i)] = dict()
            fine_json[str(i)]["samples"] = 0
            fine_json[str(i)]["datagram"] = dict()

         # 初始化临时数据存储，按1-20的具体包数量分类
        temp_data = {
            label: {f'valid_{k}': [] for k in range(1, 21)}  # valid_1 到 valid_20
            for label in content.keys()
        }

        for label, traces in content.items():   #malware
            for trace_name, trace in traces.items(): #flow_666_anon_split_00012_20200728172912
                for flow_name, flow in trace.items():   #flow_name: pcap包名_enhance_0    dateset_root+"/"+label+"/"+trace_name+"/"+flow_name[:-10]：数据流层次pcap路径
                    flow_payload = '[CLS] '
                    packet_count = 0
                    for _, packet in flow.items():
                        payload = packet[0]
                        flow_payload += payload
                        flow_payload += ' [SEP] '
                        packet_count += 1
                        if packet_count == 20:
                            break
                    # 根据包数量分类存储
                    if 1 <= packet_count <= 20:
                        temp_data[label][f'valid_{packet_count}'].append(flow_payload)
                    # if packet_count < 5:   #排除小包
                    #     continue
                    # 第二遍处理：分阶段选取数据，确保样本量尽可能达标
        for label in content.keys():
            label_idx = str(label_to_idx[label])
            fine_json[label_idx] = {"samples": 0, "datagram": {}}
            samples = []
            
            for count in range(20, 0, -1):
                samples.extend(temp_data[label][f'valid_{count}'])
                # 如果达到最小样本量则停止（如果需要限制最大数量可在此处添加判断）
                if len(samples) >= mincount:
                    break
            for i, payload in enumerate(samples, 1):  # 从1开始编号
                fine_json[label_idx]["datagram"][str(i)] = payload
            fine_json[label_idx]["samples"] = len(samples)

                    # fine_json[str(label_to_idx[label])]["samples"] += 1
                    # fine_json[str(label_to_idx[label])]["datagram"][str(fine_json[str(label_to_idx[label])]["samples"])] = flow_payload
    json_file.close()
    with open(dataset_save_file, "w") as f:
        json.dump(fine_json, f, indent=4)       #写入时会覆盖
    return dataset_save_path
    

def get_tsv(dataset_save_path):
    # 设置路径
    # dataset_name = name
    train_rate = 0.8
    samples = 500
    # dataset_save_path = f"./{fine_data}/" + dataset_name + f"_enhance{enhance}"   # 数据集保存路径
    dataset_save_path = dataset_save_path
    if not os.path.exists(dataset_save_path):
        os.makedirs(dataset_save_path)
    print("Converting to TSV format...")
    features = ["datagram"]
    dataset_extract(
        dataset_save_path=dataset_save_path,
        features=features,
        train_rate=train_rate,
        samples=samples,
    )       
    #生成的tsv文件只包含了两列，label       text_a
    #                           1      [SEP] 4500 ...      


def main(name):
    dataset_save_path = get_json(name)
    get_tsv(dataset_save_path)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # file_list = os.listdir("../../dataPcap_flow_/")
    file_list = ["cipherspectrum_aes128", "cipherspectrum_aes256", 
                 "cipherspectrum_cha", "cipherspectrum_mix", "cstent", "Edge-IIoTset"]   #wait 5
    file_list = ["itcnet"]
    # tor app ustcMAl uav_5s service idi cic-iot_300s andro_300s ustc 存在样本不平衡现象
    for file in file_list:
        print(file,"--------------------------------")
        main(file)
    '''
    1. file_list
    2. enhance
    '''