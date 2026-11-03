import os,random,json,csv
import ipaddress,pickle
from tsvAugmenter import PacketSeqAugmenter
# generate random ipv4 address
def random_ipv4():
    IPV4_MAX = ipaddress.IPv4Address._ALL_ONES
    ip_int = random.randint(0, IPV4_MAX)
    ip_str = ipaddress.IPv4Address._string_from_ip_int(ip_int)
    return ip_str

# generate random ipv6 address
def random_ipv6():
    IPV6_MAX = ipaddress.IPv6Address._ALL_ONES
    ip_int = random.randint(0, IPV6_MAX)
    ip_str = ipaddress.IPv6Address._string_from_ip_int(ip_int)
    return ip_str

def random_field(bits):
    field_max = 2**bits-1
    field_int = random.randint(0, field_max)
    return field_int

def convert_pcapng_2_pcap(pcapng_path, pcapng_file, output_path):
    
    pcap_file = output_path + pcapng_file.replace('pcapng','pcap')
    cmd = "editcap -F pcap %s %s"
    command = cmd%(pcapng_path+pcapng_file, pcap_file)
    os.system(command)
    return 0

def split_cap(pcap_file_path, output_dir, pcap_label='', split_way = 'bidirection'):
    # pcap_split_path + "splitcap/" + pcap_label + "/" + pcap_name is output
    # pcap_file_path+pcap_name is input
    split_way = "session" if split_way=='bidirection' else "flow"
    print(pcap_file_path,output_dir)
    cmd = f"mono ./SplitCap.exe -r {pcap_file_path} -s {split_way} -o {output_dir}"
    #print(cmd)
    os.system(cmd)
    return output_dir

def cut(obj, sec):      #函数是一个切片函数，用于将输入字符串按指定长度进行分割
    result = [obj[i:i+sec] for i in range(0,len(obj),sec)]      #len(obj)=122 sec=len(obj)/2=61
    try:
        remanent_count = len(result[0])%4       #remanent_count=1
    except Exception as e:
        remanent_count = 0
        print("cut datagram error!")
    if remanent_count == 0:
        pass
    else:
        result = [obj[i:i+sec+remanent_count] for i in range(0,len(obj),sec+remanent_count)]
    return result

def bigram_generation(packet_datagram, token_len = 64, flag=True): #token_len实际上是限制生成的bigram对（双字节组合）的数量
    result = ''
    generated_datagram = cut(packet_datagram,1)  #最后得到的结果就是两两一对
    token_count = 0
    for sub_string_index in range(len(generated_datagram)):
        if sub_string_index != (len(generated_datagram) - 1):
            token_count += 1
            if token_count > token_len:
                break
            else:
                merge_word_bigram = generated_datagram[sub_string_index] + generated_datagram[sub_string_index + 1]
        else:
            break
        result += merge_word_bigram
        result += ' '
    
    return result   #'||ab abcd cdef efg '

def gram_generation(packet_datagram):
    result = ''
    generated_datagram = cut(packet_datagram,2)
    for sub_string_index in range(len(generated_datagram)): #['abcd', 'efgh', 'ijkl', 'mn']
        merge_word_bigram = generated_datagram[sub_string_index]
        result += merge_word_bigram
        result += ' '
    
    return result

def size_format(size):
    # 'KB'
    file_size = '%.3f' % float(size/1000)
    return file_size

def read_data_from_json(json_data, features):
    X,Y = [], []
    for feature_index in range(len(features)):
        x = []
        for label in json_data.keys():
            x_label = []
            for sample_index in json_data[label][features[feature_index]].keys():
                x_label.append(json_data[label][features[feature_index]][sample_index])
            x.append(x_label)
            y = [label] * len(x_label)
            Y.append(y)
        X.append(x)
    return X,Y

def obtain_data(features, dataset_save_path, json_data = None):
    
    if json_data:
        X,Y = read_data_from_json(json_data,features)
    else:
        print("read dataset from json file.")
        with open(dataset_save_path + "/dataset.json","r") as f:
            dataset = json.load(f)
        X,Y = read_data_from_json(dataset,features)

    for index in range(len(X)):
        if len(X[index]) != len(Y):
            print("data and labels are not properly associated.")
            print("x:%s\ty:%s"%(len(X[index]),len(Y)))
            return -1
    return X,Y

def write_dataset_tsv(data,label,file_dir,type):
    if "train" in type:
        augmenter = PacketSeqAugmenter()
        # 先验证第一个样本的格式（避免批量报错）
        try:
            _ = augmenter.parse_seq_format(data[0])
            print("✅ 第一个样本格式验证通过")
        except Exception as e:
            raise ValueError(f"❌ 样本格式错误：{str(e)}") from e
        # 平衡并增强
        balanced_data, balanced_labels = augmenter.balance_seq_dataset(data, label)
        print(f"原始样本数: {len(data)}，增强后样本数: {len(balanced_data)}")
    else:
        balanced_data = data
        balanced_labels = label
    dataset_file = [["label", "text_a"]]
    for index in range(len(balanced_labels)):
        dataset_file.append([balanced_labels[index], balanced_data[index]])
    with open(file_dir + type + "_dataset.tsv", 'w',newline='') as f:
        tsv_w = csv.writer(f, delimiter='\t')
        tsv_w.writerows(dataset_file)
    return 0

def write_dataset_tsv_feature(data, feature,label,file_dir,type):
    dataset_file = [["label", "text_a", "feature"]]
    for index in range(len(label)):
        dataset_file.append([label[index], data[index], feature[index]])
    with open(file_dir + type + "_dataset.tsv", 'w',newline='') as f:
        tsv_w = csv.writer(f, delimiter='\t')
        tsv_w.writerows(dataset_file)
    return 0

def write_dataset_tsv_twoc(data1, data2, label,file_dir,type):
    dataset_file = [["label", "text_a", "text_b"]]
    for index in range(len(label)):
        dataset_file.append([label[index], data1[index], data2[index]])
    with open(file_dir + type + "_dataset.tsv", 'w',newline='') as f:
        tsv_w = csv.writer(f, delimiter='\t')
        tsv_w.writerows(dataset_file)
    return 0

def unlabel_data(label_data):
    nolabel_data = ""
    with open(label_data,newline='') as f:
        data = csv.reader(f,delimiter='\t')
        for row in data:
            nolabel_data += row[1] + '\n'
    nolabel_file = label_data.replace("test_dataset","nolabel_test_dataset")
    #nolabel_file = label_data.replace("train_dataset", "nolabel_train_dataset")
    with open(nolabel_file, 'w',newline='') as f:
        f.write(nolabel_data)
    return 0

# print(gram_generation("a86bad1f9bcdc49a025996f808004500003c0cc540004006b3b90a2a00d31f0d"))

def get_instance_number(file):
    count = 0
    with open(file,"rb") as f:
        try:
            while True:
                intsance = pickle.load(f)
                count+=1
                if count%1000000==0:
                    print(count)
        except EOFError:
            print(count)


def typicalsamling(group, typicalNDict):
    name = group.name
    n = typicalNDict[name]
    return group.sample(n=n)