"""
直接用第十二层的输出 
方案 A：基于码本的对比损失（推荐）
在 forward 里，对 h 加一个对比损失，让：
同位置同类的 h 靠近自己的码字,
同位置不同类的 h 远离其他类的码字

方案 B：中心损失（Center Loss）  该代码实现
维护每个 (class, position) 的 h 中心，让同类同位置的 h 靠近中心。

方案 C：类间分离正则
让不同类的中心互相远离
"""
import os
import sys
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.getcwd())
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from uer.layers import *
from uer.encoders import *
from uer.utils.constants import *
from uer.utils import *
from uer.utils.optimizers import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from uer.opts import finetune_opts
import tqdm
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from uer.config import K, d_code, vq_after_layer, L, POS
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class UsageTracker:
    def __init__(self, K, momentum=0.9, device='cpu'):
        self.K = K
        self.momentum = momentum
        self.device = device
        self.usage = torch.zeros(K, device=device)

    def update(self, indices):
        """
        indices: 任意形状，会被展平
        """
        batch_usage = torch.bincount(
            indices.reshape(-1), minlength=self.K
        ).float().to(self.device)
        self.usage = self.usage * self.momentum + batch_usage * (1 - self.momentum)

    def get_dead_mask(self, threshold_ratio=0.001):
        usage_norm = self.usage / (self.usage.sum() + 1e-8)
        avg_usage_norm = 1.0 / self.K
        # return usage_norm < threshold_ratio * avg_usage_norm
        return usage_norm < 0
    def reset_dead(self, dead_mask):
        self.usage[dead_mask] = 0.0
        self.usage[:] = 0.0

    def active_ratio(self):
        return (self.usage > 0).float().mean().item()

class EMACodebook(nn.Module):
    """
    支持 EMA 更新的 codebook。
    state_dict key 保持为 codebook.weight，兼容 nn.Embedding 预训练权重。
    """
    def __init__(self, K, d_code, decay=0.99, eps=1e-5):
        super(EMACodebook, self).__init__()
        self.K = K
        self.d_code = d_code
        self.decay = decay
        self.eps = eps
        # 保持 key 为 "weight"，与 nn.Embedding 一致
        self.weight = nn.Parameter(torch.randn(K, d_code) * 0.01)
        self.register_buffer('ema_cluster_size', torch.zeros(K))
        self.register_buffer('ema_w', self.weight.data.clone())
        # Stage 2 中 model.eval() 时仍允许更新 EMA
        self.update_ema_during_eval = False

    def forward(self, h, labels=None, num_classes=None, mask=None):
        B, P, D = h.shape
        C = num_classes
        if labels is not None:
            # 监督分配：每个位置 p 的索引 = p*C + class
            pos_offset = torch.arange(P, device=h.device).view(1, P) * C
            indices = pos_offset + labels.unsqueeze(1)          # [B, P]
        else:
            dist = torch.cdist(h, self.weight)                  # [B, P, K]
            indices = torch.argmin(dist, dim=-1)     
        z_q = F.embedding(indices, self.weight)

        if self.update_ema_during_eval:
            self._update_ema(h, indices, mask)
        # STE
        z_q = h + (z_q - h).detach()
        return z_q, indices

    def _update_ema(self, h, indices, mask=None):
        with torch.no_grad():
            if mask is not None:
                valid = mask.bool().reshape(-1)
                flat_h = h.reshape(-1, self.d_code)[valid]
                flat_indices = indices.reshape(-1)[valid]
            else:
                flat_h = h.reshape(-1, self.d_code)
                flat_indices = indices.reshape(-1)
            encodings = F.one_hot(flat_indices, self.K).float()
            # 更新每个 cluster 的计数 EMA
            self.ema_cluster_size = self.ema_cluster_size * self.decay + \
                                    (1 - self.decay) * encodings.sum(dim=0)
            # Laplace 平滑，防止 cluster size 为 0
            n = self.ema_cluster_size.sum()
            self.ema_cluster_size = (self.ema_cluster_size + self.eps) / \
                                    (n + self.K * self.eps) * n
            # 更新每个 codeword 的向量 EMA
            dw = encodings.t() @ flat_h
            self.ema_w = self.ema_w * self.decay + (1 - self.decay) * dw
            self.weight.data = self.ema_w / self.ema_cluster_size.unsqueeze(1)

    def restart_dead_codes(self, h, indices, mask=None, dead_mask=None, usage_threshold=1.0):
        with torch.no_grad():
            if mask is not None:
                valid = mask.bool().reshape(-1)
                flat_indices = indices.reshape(-1)[valid]
                flat_h = h.reshape(-1, self.d_code)[valid]
            else:
                flat_indices = indices.reshape(-1)
                flat_h = h.reshape(-1, self.d_code)

            # 如果外部传入了 dead_mask，直接使用；
            # 否则按当前 batch 的使用次数自己算。
            if dead_mask is None:
                usage = torch.bincount(flat_indices, minlength=self.K).float()
                dead_mask = usage < usage_threshold

            if dead_mask.any():
                rand_idx = torch.randint(
                    0, flat_h.size(0), (dead_mask.sum().item(),), device=h.device
                )
                self.weight.data[dead_mask] = flat_h[rand_idx]
                self.ema_w[dead_mask] = self.weight.data[dead_mask].clone()
                self.ema_cluster_size[dead_mask] = self.ema_cluster_size.mean()
                print(f"Restarted {dead_mask.sum().item()} dead codewords.")

class Classifier(nn.Module):
    """
    分类器 = BERT 编码器 + VQ 瓶颈 + 分类头。
    """
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)

        # ---- VQ 相关 ----
        self.codebook = EMACodebook(K, 768, decay=args.codebook_decay)
        self.pack_pos_embed = nn.Parameter(torch.randn(L, 768) * POS)   #位置嵌入
        self.codebook.weight.requires_grad = False  # 微调时冻结 codebook

        # ---- 分类头 ----
        self.labels_num = args.labels_num
        self.pooling = args.pooling
        self.soft_targets = args.soft_targets
        self.soft_alpha = args.soft_alpha
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, self.labels_num)

        self.commitment_coef = args.commitment_coef


        self.center_loss_coef = args.center_loss_coef   #center loss 系数
        self.center_momentum = args.center_momentum #ema动量
        self.sep_loss_coef = args.sep_loss_coef         #类间分离正则系数
        self.sep_margin = args.sep_margin               #类间距离 margin
        # 每个 (class, position) 维护一个 h 中心
        self.register_buffer(
            'h_centers',
            torch.zeros(self.labels_num, L, args.hidden_size)
        )

    def quantize(self, h, labels=None, mask=None):
        z_q, indices = self.codebook(
            h, labels=labels, num_classes=self.labels_num, mask=mask
        )
        return z_q, indices

    def forward(self, src, tgt, seg):
        # ---- Embedding ----
        emb = self.embedding(src, seg)  # [B, L, H]
        hidden = emb
        batch_size, seq_length, _ = hidden.size()

        # ---- 构造 attention mask（与 TransformerEncoder 一致）----
        mask = (seg > 0).unsqueeze(1).repeat(1, seq_length, 1).unsqueeze(1).float()
        mask = (1.0 - mask) * -10000.0
        position_bias = None

        # ---- Lower Encoder：VQ 之前 ----
        for i in range(self.encoder.layers_num):
            hidden, _ = self.encoder.transformer[i](hidden, mask, position_bias=position_bias)

        # ---- VQ 处理：只取 [PACK] token 位置 ----
        # pack_list = [11, 22, ..., 220]，对应每个包的第一个 token 位置
        pack_list = [i * 11 for i in range(0, 21)]
        pack_hidden = hidden[:, pack_list, :]          # [B, 20, H]
        pack_mask = (seg[:, pack_list] >= 0).float()   # [B, 20]

        h = pack_hidden                  # [B, 20, d_code]

        pos_embed = self.pack_pos_embed[:pack_hidden.size(1)].unsqueeze(0)  # [1, L, d_code]
        if POS > 0:
            h = h + pos_embed
        z_q, indices = self.quantize(h, labels=tgt, mask=pack_mask)     # [B, 20, d_code], [B, 20]
        # hidden[:, pack_list, :] = z_q

        # ---- 新增：计算每个样本的重构误差（用于 OOD 拒识）----
        with torch.no_grad():
            recon = ((pack_hidden - z_q) ** 2).mean(dim=-1)           # [B, L]
            recon_per_sample = (recon * pack_mask).sum(dim=-1) / (pack_mask.sum(dim=-1) + 1e-8)  # [B]
            # recon_per_sample = recon.max(dim=-1)[0]

        # ---- 分类头：取 [CLS] 向量 ----
        output = hidden[:, 0, :]                     # [CLS]
        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)             # [B, labels_num]

        # ---- 损失计算 ----
        loss = None
        cls_loss = torch.tensor(0.0, device=logits.device)
        if tgt is not None:
            orth_loss = 0
            cls_loss = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            loss = cls_loss
            e_k = F.embedding(indices, self.codebook.weight).detach()
            commit_loss = (((h - e_k) ** 2).mean(dim=-1) * pack_mask).sum() / (pack_mask.sum() + 1e-8)
            
            # ---- Center Loss ----
            with torch.no_grad():
                # EMA 更新 h_centers
                for b in range(h.size(0)):
                    c = tgt[b].item()
                    valid = pack_mask[b].bool()
                    if valid.any():
                        self.h_centers[c, valid] = (
                            self.center_momentum * self.h_centers[c, valid] +
                            (1 - self.center_momentum) * h[b, valid].detach()
                        )
            
            # 计算当前 h 到对应中心的距离
            centers = self.h_centers[tgt]  # [B, L, D]
            center_loss = (((h - centers) ** 2) * pack_mask.unsqueeze(-1)).sum() / (pack_mask.sum() + 1e-8)

            # ---- 方案 C：类间分离正则 ----
            # 同位置上，把 h 推离"最近的其他类中心"，距离超过 margin 不再惩罚
            all_centers = self.h_centers.detach().permute(1, 0, 2)        # [L, C, D]
            # h: [B, L, 1, D]  vs  all_centers: [1, L, C, D]
            dist = ((h.unsqueeze(2) - all_centers.unsqueeze(0)) ** 2).sum(dim=-1)  # [B, L, C]
            # 屏蔽自己所属的类，只保留其他类的中心
            own_idx = tgt.view(-1, 1, 1).expand(-1, h.size(1), 1)         # [B, L, 1]
            dist.scatter_(2, own_idx, float('inf'))
            min_other_dist = dist.min(dim=-1).values                      # [B, L]
            sep_loss = (F.relu(self.sep_margin - min_other_dist) * pack_mask).sum() \
                       / (pack_mask.sum() + 1e-8)
            
            loss = cls_loss + self.commitment_coef * commit_loss \
                    + self.center_loss_coef * center_loss \
                    + self.sep_loss_coef * sep_loss
            h_div_loss = center_loss
            div_orth = [h_div_loss, sep_loss]
            # codebook 已冻结，.detach() 是显式保险
            return loss, logits, indices, h, z_q, pack_mask, cls_loss, div_orth, recon_per_sample
        else:
            return None, logits, indices, h, z_q, pack_mask, None, None, recon_per_sample
def resize_codebook(model, new_K,new_d_code=None, device='cuda'):
    """
    在 Stage 2 之前把 codebook 扩容到 new_K。
    直接重新创建 EMACodebook，保持 state_dict key 不变。
    """
    old_decay = model.codebook.decay
    d_code = new_d_code if new_d_code is not None else model.codebook.d_code
    model.codebook = EMACodebook(new_K, d_code, decay=old_decay).to(device)
    model.codebook.weight.requires_grad = False
    # model.K = new_K
    print(f"Resized codebook from old size to {new_K}")

def initialize_codebook_from_dev_h(args, model, src, tgt, seg):
    """
    用 dev 集上每个类、每个位置的 h 平均值来初始化 codebook。
    填充顺序：
        第0个位置: [class0_avg, class1_avg, ..., class(C-1)_avg]
        第1个位置: [class0_avg, class1_avg, ..., class(C-1)_avg]
        ...
        第(L-1)个位置: [class0_avg, ..., class(C-1)_avg]
    其中 L = 20 为 [PACK] 数量，C = labels_num 为已知类数。

    若 K > C * L，则剩余 codeword 保持原初始化；
    若 K < C * L，则只填充前 K 个。
    """
    print("Initializing codebook from train set class-position h averages...")
    model.eval()

    # src = torch.LongTensor([s[0] for s in devset])
    # tgt = torch.LongTensor([s[1] for s in devset])
    # seg = torch.LongTensor([s[2] for s in devset])

    C = args.labels_num
    # L = 20
    d = 768
    # global K
    # new_K = int((C*L)*(1+0.05))
    new_K = int((C*L)*(1))
    resize_codebook(model, new_K, d,args.device)
    # 累加每个 (class, position) 的有效 h
    h_sums = torch.zeros(C, L, d, device=args.device)
    h_counts = torch.zeros(C, L, device=args.device, dtype=torch.long)

    batch_size = args.test_batch_size
    with torch.no_grad():
        for src_batch, tgt_batch, seg_batch in batch_loader(batch_size, src, tgt, seg):
            src_batch = src_batch.to(args.device)
            tgt_batch = tgt_batch.to(args.device)
            seg_batch = seg_batch.to(args.device)

            # 前向到 VQ 层，获取 h
            _, _, indices, h, _, pack_mask, _, _, _ = model(src_batch, None, seg_batch)
            # h: [B, L, d], pack_mask: [B, L]

            valid_mask = pack_mask.bool()  # [B, L] 512,20

            # 按位置分别累加
            for pos in range(L):
                pos_valid = valid_mask[:, pos]
                if not pos_valid.any():
                    continue

                pos_h = h[:, pos, :][pos_valid]      # [B_pos, d]   
                pos_labels = tgt_batch[pos_valid]     # [B_pos] 

                for c in range(C):
                    mask_c = pos_labels == c
                    if mask_c.any():
                        h_sums[c, pos] += pos_h[mask_c].sum(dim=0)
                        h_counts[c, pos] += mask_c.sum().item()

    # 计算平均值
    eps = 1e-8
    h_avgs = h_sums / (h_counts.unsqueeze(-1).float() + eps)  # [C, L, d]

    # 按 (position, class) 顺序展平
    codebook_init = []
    for pos in range(L):
        for c in range(C):
            codebook_init.append(h_avgs[c, pos])
    codebook_init = torch.stack(codebook_init, dim=0).to(args.device)  # [C*L, d]

    # 填充 codebook
    num_init = min(codebook_init.size(0), model.codebook.K)
    model.codebook.weight.data[:num_init] = codebook_init[:num_init]

    # 同步 EMA 缓冲区，避免初始化后立即被 EMA “洗掉”
    if isinstance(model.codebook, EMACodebook):
        model.codebook.ema_w.data[:num_init] = codebook_init[:num_init].clone()
    print(f"Initialized {num_init}/{model.codebook.K} codewords from "
          f"({C} classes x {L} positions = {C*L}).")

def count_labels_num(path):
    labels_set, columns = set(), {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
            line = line.strip().split("\t")
            label = int(line[columns["label"]])
            labels_set.add(label)
    return len(labels_set)


def load_or_initialize_parameters(args, model):
    if args.pretrained_model_path is not None and os.path.exists(args.pretrained_model_path):
        print(f"Initialize with pretrained model. {args.pretrained_model_path}")
        # 多卡训练保存的权重可能在 cuda:1/2/3，统一映射到当前可见的 cuda:0
        state = torch.load(args.pretrained_model_path, weights_only=False,
                    map_location={'cuda:1':'cuda:0','cuda:2':'cuda:0','cuda:3':'cuda:0'})
            # 过滤掉 codebook / proj_in / proj_out
        filtered = {k: v for k, v in state.items()
                    if not any(s in k for s in ['codebook', 'proj_in', 'proj_out'])}
        del state   # 新增：原始 checkpoint 立即释放，不再驻留
        model.load_state_dict(filtered, strict=False)
        del filtered  # 新增：过滤副本释放
    else:
        print("Initialize with normal distribution.")
        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)

def build_optimizer(args, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    if args.optimizer in ["adamw"]: #default
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    else:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate,
                                                  scale_parameter=False, relative_step=False)
    if args.scheduler in ["constant"]:
        scheduler = str2scheduler[args.scheduler](optimizer)
    elif args.scheduler in ["constant_with_warmup"]:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps * args.warmup)
    else:       #linear  default
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps * args.warmup, args.train_steps)
    return optimizer, scheduler


def batch_loader(batch_size, src, tgt, seg):
    instances_num = src.size()[0]
    for i in range(instances_num // batch_size):
        src_batch = src[i * batch_size: (i + 1) * batch_size, :]
        tgt_batch = tgt[i * batch_size: (i + 1) * batch_size]
        seg_batch = seg[i * batch_size: (i + 1) * batch_size, :]

        yield src_batch, tgt_batch, seg_batch
    if instances_num > instances_num // batch_size * batch_size:
        src_batch = src[instances_num // batch_size * batch_size:, :]
        tgt_batch = tgt[instances_num // batch_size * batch_size:]
        seg_batch = seg[instances_num // batch_size * batch_size:, :]

        yield src_batch, tgt_batch, seg_batch
def dataset_to_tensors(dataset):
    """把数据集列表一次性转成 CPU 张量，之后可释放原始 Python 列表。"""
    src = torch.LongTensor([s[0] for s in dataset])
    tgt = torch.LongTensor([s[1] for s in dataset])
    seg = torch.LongTensor([s[2] for s in dataset])
    return src, tgt, seg

def read_dataset(args, path):
    """读取 tsv 格式微调数据集：text_a \t label"""
    dataset, columns = [], {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
            line = line[:-1].split("\t")
            tgt = int(line[columns["label"]])
            if args.soft_targets and "logits" in columns.keys():
                soft_tgt = [float(value) for value in line[columns["logits"]].split(" ")]
            text = line[columns["text_a"]]
            text = [t for t in text.split()]
            src = args.tokenizer.convert_tokens_to_ids(text)
            seg = [1]
            i = 1
            while True:
                ip_seg = [i] * 7
                i += 1
                seg.extend(ip_seg)
                tcp_seg = [i] * 4
                i += 1
                seg.extend(tcp_seg)
                if len(seg) == len(src):
                    break

            if len(src) > args.seq_length:
                src = src[: args.seq_length]
                seg = seg[: args.seq_length]
            while len(src) < args.seq_length:
                src.append(0)
                seg.append(0)
            dataset.append((src, tgt, seg))
    return dataset


def split_known_unknown_classes(args, trainset_raw, devset_raw, testset_raw):
    """
    按类别划分已知类 / 未知类。
    - train/dev：只保留已知类，标签重映射为 0~K-1
    - test：已知类重映射为 0~K-1，未知类标签设为 -1
    Returns:
        trainset, devset, testset, known_classes, label_map
    """
    # 从训练集获取所有类别
    all_train_labels = sorted(set(s[1] for s in trainset_raw))
    total_classes = len(all_train_labels)

    # 确定未知类数量
    if args.unknown_classes_num > 0:
        num_unknown = min(args.unknown_classes_num, total_classes - 1)
    else:
        num_unknown = int(total_classes * args.unknown_classes_ratio)
    num_unknown = max(0, min(num_unknown, total_classes - 1))  # 至少保留 1 个已知类

    # 随机选择未知类（使用 args.seed 保证可复现）
    rng = random.Random(args.seed)
    unknown_classes = set(rng.sample(all_train_labels, num_unknown))
    known_classes = [l for l in all_train_labels if l not in unknown_classes]

    print(f"Total classes in train: {total_classes}")
    print(f"Known classes ({len(known_classes)}): {known_classes}")
    print(f"Unknown classes ({len(unknown_classes)}): {sorted(unknown_classes)}")

    # 已知类标签重映射
    label_map = {old: new for new, old in enumerate(known_classes)}

    # 过滤并 remap 训练集 / 验证集
    trainset = [(src, label_map[tgt], seg) for src, tgt, seg in trainset_raw if tgt in label_map]
    devset = [(src, label_map[tgt], seg) for src, tgt, seg in devset_raw if tgt in label_map]

    # 测试集：已知类 remap，未知类标记为 -1
    testset = []
    for src, tgt, seg in testset_raw:
        if tgt in label_map:
            testset.append((src, label_map[tgt], seg))
        else:
            testset.append((src, -1, seg))

    return trainset, devset, testset, known_classes, label_map


def train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch):
    model.zero_grad()

    src_batch = src_batch.to(args.device)
    tgt_batch = tgt_batch.to(args.device)
    seg_batch = seg_batch.to(args.device)

    # 训练时返回 VQ 中间量，用于计算/打印 commitment loss
    loss, _, indices, h, _, pack_mask, cls_loss, div_orth, _ = model(src_batch, tgt_batch, seg_batch)
    loss.backward()
    optimizer.step()
    scheduler.step()

    return loss.item(), cls_loss.item(), div_orth, indices, h, pack_mask

def build_class_sequence_sets(args, model, src, tgt, seg, pad_value=-1, max_len=L):
    """
    收集每个类出现过的所有唯一 codeword 序列。
    序列会按有效长度补齐/截断到 max_len，用 pad_value 填充。

    Returns:
        class_sequences: dict {label: set(tuple)}
    """
    print("Building class sequence sets...")
    model.eval()
    class_sequences = {c: set() for c in range(args.labels_num)}
    # src = torch.LongTensor([s[0] for s in dataset])
    # tgt = torch.LongTensor([s[1] for s in dataset])
    # seg = torch.LongTensor([s[2] for s in dataset])
    batch_size = args.test_batch_size
    with torch.no_grad():
        for src_batch, tgt_batch, seg_batch in batch_loader(batch_size, src, tgt, seg):
            src_batch = src_batch.to(args.device)
            tgt_batch = tgt_batch.to(args.device)
            seg_batch = seg_batch.to(args.device)
            _, _, indices, _, _, pack_mask, _, _, _ = model(src_batch, None, seg_batch)
            for b in range(indices.size(0)):
                label = tgt_batch[b].item()
                valid_len = int(pack_mask[b].sum().item())
                # 只取有效位置
                seq = indices[b, :valid_len].cpu().tolist()
                seq = seq + [pad_value] * (max_len - valid_len)
                seq = tuple(seq)
                class_sequences[label].add(seq) #如果不存在就添加

    for c, seqs in class_sequences.items():
        print(f"Class {c}: {len(seqs)} unique sequences")
    return class_sequences

def build_class_sequence_thresholds(args, model, src, tgt, seg, class_sequences,
                                    metric='hamming', pad_value=-1, max_len=L,
                                    percentile=95.0, min_count=1):
    """
    对 dataset 中每个样本，找到其真实类在训练集中最近的码字序列，
    把该样本的重构误差记录到该序列下，最后每个序列取分位数作为专属阈值。
    """
    print("Building per-sequence reconstruction thresholds...")
    model.eval()

    # src = torch.LongTensor([s[0] for s in dataset])
    # tgt = torch.LongTensor([s[1] for s in dataset])
    # seg = torch.LongTensor([s[2] for s in dataset])

    # class -> {seq_tuple: [recon1, recon2, ...]}
    seq_recons = {c: {} for c in class_sequences.keys()}
    class_recons = {c: [] for c in class_sequences.keys()}
    batch_size = args.test_batch_size

    with torch.no_grad():
        for src_batch, tgt_batch, seg_batch in batch_loader(batch_size, src, tgt, seg):
            src_batch = src_batch.to(args.device)
            tgt_batch = tgt_batch.to(args.device)
            seg_batch = seg_batch.to(args.device)

            _, _, indices, _, _, pack_mask, _, _, recon_per_sample = model(src_batch, None, seg_batch)

            for b in range(indices.size(0)):
                true_class = tgt_batch[b].item()
                train_seqs = class_sequences[true_class]
                if len(train_seqs) == 0:
                    continue

                valid_len = int(pack_mask[b].sum().item())
                seq = indices[b, :valid_len].cpu().tolist()
                seq = seq + [pad_value] * (max_len - valid_len)
                seq_tuple = tuple(seq)

                # 找训练集中最近的序列
                min_dist, nearest_seq = min(
                    ((sequence_distance(seq_tuple, train_seq, metric, pad_value), train_seq)
                     for train_seq in train_seqs),
                    key=lambda x: x[0]
                )

                # 把当前样本的 recon 挂到最近序列下
                seq_recons[true_class].setdefault(nearest_seq, []).append(recon_per_sample[b].item())
                class_recons[true_class].append(recon_per_sample[b].item())
    # 计算每个序列的阈值
    class_global_threshold = {}
    for c, recons in class_recons.items():
        if len(recons) > 0:
            class_global_threshold[c] = float(np.percentile(recons, percentile))
        else:
            class_global_threshold[c] = float('inf')
    seq_thresholds = {c: {} for c in class_sequences.keys()}
    for c, seq_dict in seq_recons.items():
        for seq, recons in seq_dict.items():
            if len(recons) >= min_count:
                seq_thresholds[c][seq] = float(np.percentile(recons, percentile))
            else:
                # 样本太少，不依赖 recon，只让序列距离起作用
                # seq_thresholds[c][seq] = float('inf')
                seq_thresholds[c][seq] = class_global_threshold.get(c, float('inf'))

    print(f"Built recon thresholds for {sum(len(v) for v in seq_thresholds.values())} sequences.")
    return seq_thresholds, class_global_threshold

def sequence_distance(seq1, seq2, metric='hamming', pad_value=-1):
    """
    计算两个等长序列的距离。
    seq1, seq2: tuple/list of length max_len
    """
    if metric == 'hamming':
        # 统计非 padding 位置中 codeword 不同的数量
        dist = 0
        for i, (a, b) in enumerate(zip(seq1, seq2)):
            if a != b:
                dist += 1 if i == 0 else 1
        return dist
    elif metric == 'edit':
        # Levenshtein 编辑距离，允许插入/删除/替换
        return levenshtein_distance(seq1, seq2)
    elif metric == 'jaccard':
        # n-gram Jaccard 距离
        n = 2
        set1 = set(tuple(seq1[i:i+n]) for i in range(len(seq1) - n + 1) if pad_value not in seq1[i:i+n])
        set2 = set(tuple(seq2[i:i+n]) for i in range(len(seq2) - n + 1) if pad_value not in seq2[i:i+n])
        if len(set1 | set2) == 0:
            return 0.0
        return 1.0 - len(set1 & set2) / len(set1 | set2)
    else:
        raise ValueError(f"Unknown sequence distance metric: {metric}")

def levenshtein_distance(seq1, seq2):
    """Levenshtein 编辑距离"""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if seq1[i-1] == seq2[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # deletion
                dp[i][j-1] + 1,      # insertion
                dp[i-1][j-1] + cost  # substitution
            )
    return dp[m][n]

def calibrate_sequence_threshold(args, model, src, tgt, seg, class_sequences,
                                  metric='hamming', pad_value=-1, max_len=L):
    """
    对 dev 集中每个样本，计算其到"真实类"最近训练序列的距离。
    取 95% 分位数作为拒识阈值。
    """
    print("Calibrating sequence rejection thresholds...")
    model.eval()
    # src = torch.LongTensor([s[0] for s in devset])
    # tgt = torch.LongTensor([s[1] for s in devset])
    # seg = torch.LongTensor([s[2] for s in devset])
    class_scores = {c: [] for c in class_sequences.keys()}
    batch_size = args.test_batch_size
    with torch.no_grad():
        for src_batch, tgt_batch, seg_batch in batch_loader(batch_size, src, tgt, seg):
            src_batch = src_batch.to(args.device)
            tgt_batch = tgt_batch.to(args.device)
            seg_batch = seg_batch.to(args.device)
            _, logits, indices, _, _, pack_mask, _, _, _ = model(src_batch, None, seg_batch)
            pred_cls = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
            for b in range(indices.size(0)):
                true_class = tgt_batch[b].item()  # 用真实标签
                valid_len = int(pack_mask[b].sum().item())
                seq = indices[b, :valid_len].cpu().tolist()
                seq = seq + [pad_value] * (max_len - valid_len)
                # 到真实类最近训练序列的最小距离
                train_seqs = class_sequences[true_class]
                if len(train_seqs) == 0:
                    min_dist = max_len
                else:
                    min_dist = min(
                        sequence_distance(seq, train_seq, metric, pad_value)
                        for train_seq in train_seqs
                    )
                class_scores[true_class].append(min_dist)
    percentile = args.rejection_percentile
    thresholds = {}
    for c, scores in class_scores.items():
        if len(scores) > 0:
            thresholds[c] = float(np.percentile(scores, percentile))
        else:
            thresholds[c] = max_len  # 保守阈值
        print(f"Class {c} sequence threshold at {percentile}% percentile: {thresholds[c]:.6f} "
              f"(based on {len(scores)} samples with true label class {c})")
    return thresholds, class_scores


def evaluate(args, src, tgt, seg, class_sequences=None,
             seq_threshold=None, seq_metric='hamming',seq_recon_thresholds=None,
             print_confusion_matrix=False, open_set=False,
             class_global_recon_threshold=None):
    """
    模型评估。如果传入 fingerprints + threshold + p_bg，则启用拒识机制，
    将 JS 散度超过阈值的样本判为未知（标签 -1）。
    支持测试集中出现训练时未见过的未知类（true_label == -1）。
    """
    # src = torch.LongTensor([sample[0] for sample in dataset])
    # tgt = torch.LongTensor([sample[1] for sample in dataset])
    # seg = torch.LongTensor([sample[2] for sample in dataset])

    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)
    y_true, y_pred = [], []
    args.model.eval()

    rejected = 0

    # 开放集统计量
    known_correct = 0      # 已知类判对
    known_total = 0        # 已知类总数
    unknown_correct_rej = 0   # 未知类正确拒识
    unknown_total = 0         # 未知类总数
    false_accept = 0          # 未知类被错分为已知类
    false_reject = 0          # 已知类被错分为未知类
   
    batch_size = args.test_batch_size
    for i, (src_batch, tgt_batch, seg_batch) in enumerate(batch_loader(batch_size, src, tgt, seg)):
        src_batch = src_batch.to(args.device)
        tgt_batch = tgt_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)
        with torch.no_grad():
            _, logits, indices, _, _, pack_mask, _, _, recon_per_sample = args.model(src_batch, None, seg_batch)
        pred_cls = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
        gold = tgt_batch

        for b in range(pred_cls.size(0)):
            true_label = gold[b].item()
            pred_label = pred_cls[b].item()
            predicted_class = pred_label

            if open_set:
                seq_reject = False
                recon_reject = False
                nearest_seq = None

                if class_sequences is not None and seq_threshold is not None:
                    valid_len = int(pack_mask[b].sum().item())
                    seq = indices[b, :valid_len].cpu().tolist()
                    seq = seq + [-1] * (L - valid_len)
                    seq_tuple = tuple(seq)
                    train_seqs = class_sequences[predicted_class]

                    if len(train_seqs) > 0:
                        # 找最近的训练序列及其距离
                        min_dist, nearest_seq = min(
                            ((sequence_distance(seq_tuple, train_seq, seq_metric), train_seq)
                             for train_seq in train_seqs),
                            key=lambda x: x[0]
                        )

                        seq_th = seq_threshold
                        if isinstance(seq_threshold, dict):
                            seq_th = seq_threshold.get(predicted_class, L)
                        seq_reject = min_dist > seq_th

                        # 序列距离不超，再判该序列下的重构误差
                        if (not seq_reject) and (seq_recon_thresholds is not None):
                            per_seq_th = seq_recon_thresholds[predicted_class].get(nearest_seq, float('inf'))
                            if per_seq_th is None or per_seq_th == float('inf'):
                                per_seq_th = class_global_recon_threshold.get(predicted_class, float('inf'))
                            recon_reject = recon_per_sample[b].item() > per_seq_th

                if not (seq_reject or recon_reject) and true_label == -1:
                    print(predicted_class, true_label, min_dist, seq,
                          recon_per_sample[b].item() if recon_per_sample is not None else 0,
                          per_seq_th)

                if seq_reject or recon_reject:
                    pred_label = -1
                    rejected += 1

            y_true.append(true_label)
            y_pred.append(pred_label)

            # 开放集统计
            is_known = (true_label != -1)
            is_rejected = (pred_label == -1)

            if is_known:
                known_total += 1
                if is_rejected:
                    false_reject += 1
                else:
                    # 只把已知-已知对放入 confusion
                    if 0 <= true_label < args.labels_num and 0 <= pred_label < args.labels_num:
                        confusion[pred_label, true_label] += 1
                    if pred_label == true_label:
                        known_correct += 1
            else:
                unknown_total += 1
                if is_rejected:
                    unknown_correct_rej += 1
                else:
                    false_accept += 1

    # ---- 打印指标 ----
    print("Acc. (Correct/Total): {:.4f} ({}/{})".format(
        known_correct / max(known_total, 1), known_correct, known_total))
    if open_set:
        print(f"Rejected as unknown: {rejected}/{len(y_pred)} ({100.0 * rejected / len(y_pred):.2f}%)")
        print(f"Known class accuracy: {known_correct}/{known_total} ({100.0 * known_correct / max(known_total, 1):.2f}%)")
        print(f"Unknown detection rate: {unknown_correct_rej}/{unknown_total} ({100.0 * unknown_correct_rej / max(unknown_total, 1):.2f}%)")
        print(f"False accept (unknown -> known): {false_accept}")
        print(f"False reject (known -> unknown): {false_reject}")

    if print_confusion_matrix and known_total > 0:
        print("Confusion matrix (known classes only):")
        torch.set_printoptions(threshold=1000000, edgeitems=1000000, linewidth=1000000)
        print(confusion)
        print("Report precision, recall, and f1:")
        eps = 1e-9
        for i in range(confusion.size(0)):
            p = confusion[i, i].item() / (confusion[i, :].sum().item() + eps)
            r = confusion[i, i].item() / (confusion[:, i].sum().item() + eps)
            if (p + r) == 0:
                f1 = 0
            else:
                f1 = 2 * p * r / (p + r)
            print("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i, p, r, f1))

    # 仅对已知类计算 sklearn 指标
    y_true_known = [t for t, p in zip(y_true, y_pred) if p >= 0 and t >= 0]
    y_pred_known = [p for t, p in zip(y_true, y_pred) if p >= 0 and t >= 0]
    if len(y_true_known) > 0:
        print("Macro precision: {:.4f}, Micro precision: {:.4f}, Weighted precision: {:.4f}".format(
            precision_score(y_true_known, y_pred_known, average='macro'),
            precision_score(y_true_known, y_pred_known, average='micro'),
            precision_score(y_true_known, y_pred_known, average='weighted')))
        print("Macro recall: {:.4f}, Micro recall: {:.4f}, Weighted recall: {:.4f}".format(
            recall_score(y_true_known, y_pred_known, average='macro'),
            recall_score(y_true_known, y_pred_known, average='micro'),
            recall_score(y_true_known, y_pred_known, average='weighted')))
        print("Macro f1: {:.4f}, Micro f1: {:.4f}, Weighted f1: {:.4f}".format(
            f1_score(y_true_known, y_pred_known, average='macro'),
            f1_score(y_true_known, y_pred_known, average='micro'),
            f1_score(y_true_known, y_pred_known, average='weighted')))
    return {
        'f1': f1_score(y_true_known, y_pred_known, average='macro') if len(y_true_known) > 0 else 0.0,
        'acc': accuracy_score(y_true_known, y_pred_known) if len(y_true_known) > 0 else 0.0,
        'pr': precision_score(y_true_known, y_pred_known, average='macro') if len(y_true_known) > 0 else 0.0,
        'rc': recall_score(y_true_known, y_pred_known, average='macro') if len(y_true_known) > 0 else 0.0,
        'confusion': confusion,
        'rejected': rejected,
        'known_total': known_total,
        'known_correct': known_correct,
        'unknown_total': unknown_total,
        'unknown_correct_rej': unknown_correct_rej,
        'false_accept': false_accept,
        'false_reject': false_reject,
    }

def main(datasetname="cipherspectrum_aes128"):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    finetune_opts(parser)
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")
    parser.add_argument("--earlystop", type=int, default=500, help="early stop rounds.")
    parser.add_argument("--samples", type=int, default=500, help="samples.")
    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                        help="Specify the tokenizer.")
    parser.add_argument("--soft_targets", action='store_true',
                        help="Train model with logits.")
    parser.add_argument("--soft_alpha", type=float, default=0.5,
                        help="Weight of the soft targets loss.")
    parser.add_argument("--is_moe", action="store_true", help="adopt moe layer.")


    # ---- 新增：VQ / 拒识相关超参数 ----
    parser.add_argument("--commitment_coef", type=float, default=0.00,      #去除这个损失
                        help="Commitment loss 系数，0 表示不使用。推荐 0.01~0.1。")

    parser.add_argument("--no_rejection", type=bool, default=False,
                        help="禁用拒识机制，只做普通分类评估。")

    # ---- 新增：开放集已知/未知类划分 ----
    parser.add_argument("--unknown_classes_num", type=int, default=5,       #设置为0就闭集分类
                        help="指定从训练集中选多少个类别作为未知类（优先级高于 ratio）。")
    parser.add_argument("--unknown_classes_ratio", type=float, default=0.0,
                        help="指定训练集中多少比例类别作为未知类（0.0~1.0）。")
    parser.add_argument("--test_batch_size", type=int, default=512, help="samples.")
    # Stage 2 相关
    parser.add_argument("--codebook_decay", type=float, default=0.99,
                        help="EMA 衰减系数。")
    parser.add_argument("--use_recon_rejection", type=bool, default=True,
                        help="使用重构误差作为额外的 OOD 拒识信号。")

    parser.add_argument("--center_loss_coef", type=float, default=0.01,
                        help="Center loss 系数，推荐 0.001~0.05。")
    parser.add_argument("--center_momentum", type=float, default=0.9,
                        help="h_centers 的 EMA 动量，推荐 0.9~0.99。")
     # ---- 方案 C：类间分离正则 ----
    parser.add_argument("--sep_loss_coef", type=float, default=0.01,
                        help="类间分离正则系数，推荐 0.001~0.05，0 表示关闭。")
    parser.add_argument("--sep_margin", type=float, default=10.0,
                        help="类间最小平方距离 margin，大于该距离不再惩罚。")

    parser.add_argument("--rejection_percentile", type=float, default=95.0,
                        help="验证集上标定拒识阈值的分位数，默认 95%。")
    parser.add_argument("--recon_percentile", type=float, default=95.0,
                        help="验证集上标定重构误差阈值的分位数。")
    parser.add_argument("--datasets", type=str, default=None,
                        help="逗号分隔的数据集列表，仅由父进程传入，main 内部不使用。")
    args = parser.parse_args()

    # 加载模型超参
    args = load_hyperparam(args)
    set_seed(args.seed)
    args.batch_size = 32
    args.epochs_num = 100
    args.test_path = f"fine-data/fine_data_flow/{datasetname}_enhance1/0.8/test_500_dataset.tsv"
    args.train_path = f"fine-data/fine_data_flow/{datasetname}_enhance1/0.8/train_500_dataset.tsv"
    args.dev_path = f"fine-data/fine_data_flow/{datasetname}_enhance1/0.8/valid_500_dataset.tsv"

    # 构建 tokenizer
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # ---- 正常训练流程 ----
    # 读取原始训练/验证/测试集
    trainset_raw = read_dataset(args, args.train_path)
    devset_raw = read_dataset(args, args.dev_path)
    testset_raw = read_dataset(args, args.test_path) if args.test_path is not None else []

    # 划分已知类 / 未知类
    if args.unknown_classes_num > 0 or args.unknown_classes_ratio > 0.0:
        trainset, devset, testset, known_classes, label_map = split_known_unknown_classes(
            args, trainset_raw, devset_raw, testset_raw)
    else:
        # 不划分，全部作为已知类（闭集）
        all_labels = sorted(set(s[1] for s in trainset_raw))
        label_map = {old: new for new, old in enumerate(all_labels)}
        trainset = [(src, label_map[tgt], seg) for src, tgt, seg in trainset_raw]
        devset = [(src, label_map[tgt], seg) for src, tgt, seg in devset_raw]
        testset = [(src, label_map.get(tgt, -1), seg) for src, tgt, seg in testset_raw]
        known_classes = all_labels
        print(f"No unknown class split. All {len(known_classes)} classes treated as known.")

    args.labels_num = len(known_classes)
    print(f"Final labels_num (known classes): {args.labels_num}")

    # 构建模型
    model = Classifier(args)
    load_or_initialize_parameters(args, model)
    loss_name = "/loss_open.txt"
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(args.device)
    loss_save_path = args.dev_path.rsplit('/', 1)[0] + "/nofeature_" + str(args.samples) + "_" + args.pretrained_model_path.rsplit('/', 1)[1].rsplit('-', 1)[1]
    if not os.path.exists(loss_save_path):
        os.makedirs(loss_save_path)
    with open(loss_save_path + loss_name, "a") as f:
        f.write(f"\n\npos_odd12_center_sep   {str(POS)}\n")
        if args.unknown_classes_num == 0 and args.unknown_classes_ratio == 0.0:
            f.write("Closed-set classification\n")
            args.no_rejection = True
        else:
            args.no_rejection = False
    # ---- 训练阶段 ----
    random.shuffle(trainset)
    instances_num = len(trainset)
    batch_size = args.batch_size
    src, tgt, seg = dataset_to_tensors(trainset)
    dev_src, dev_tgt, dev_seg = dataset_to_tensors(devset)
    test_src, test_tgt, test_seg = dataset_to_tensors(testset)
    # src = torch.LongTensor([example[0] for example in trainset])
    # tgt = torch.LongTensor([example[1] for example in trainset])
    # seg = torch.LongTensor([example[2] for example in trainset])
    del trainset, devset, testset, trainset_raw, devset_raw, testset_raw

    # args.train_steps = int(instances_num * 100 / batch_size) + 1
    args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    print("Batch size: ", batch_size)
    print("The number of training instances:", instances_num)
    # 新增：在训练前先初始化 codebook
    # initialize_codebook_from_dev_h(args, model, trainset)
    initialize_codebook_from_dev_h(args, model, src, tgt, seg)

    # 用 codebook 初始化 h_centers: [C*L, D] -> [L, C, D] -> [C, L, D]
    with torch.no_grad():
        C = args.labels_num
        D = args.hidden_size
        model.h_centers.copy_(
            model.codebook.weight.view(L, C, D).permute(1, 0, 2).contiguous()
        )

    optimizer, scheduler = build_optimizer(args, model)
    args.model = model
    total_loss, total_cls_loss, total_div, total_orth  = 0.0, 0.0, 0.0, 0.0
    best_result, best_result_round = 0.0, 0
    args.output_model_path = os.path.join(loss_save_path, "best_model_pos_ood12.bin")
    best_model_path = os.path.join(loss_save_path, "best_model_pos_ood12.bin")
    if not os.path.exists(args.output_model_path):
        print("Stage 1: Fine-tuning classifier...")
        model.codebook.update_ema_during_eval = True
        # tracker = UsageTracker(model.codebook.K, momentum=0.9, device=args.device)
        for epoch in tqdm.tqdm(range(1, args.epochs_num + 1)):
            model.train()
            for i, (src_batch, tgt_batch, seg_batch) in \
                    enumerate(batch_loader(batch_size, src, tgt, seg)):
                loss, cls_loss, div_orth, indices, h, pack_mask = train_model(
                    args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch)
                
                # with torch.no_grad():
                    # valid_indices = indices[pack_mask.bool()]
                    # tracker.update(valid_indices)
                total_loss += loss
                total_cls_loss += cls_loss
                total_div += div_orth[0]
                total_orth += div_orth[1]

                lr = optimizer.param_groups[0]['lr']
                if (i + 1) % args.report_steps == 0:
                    print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}, "
                        "cls_loss: {:.3f}, center_loss: {:.3f}, orth_loss: {:.3f}, lr: {:.3e}".format(
                            epoch, i + 1, total_loss / args.report_steps,
                            total_cls_loss / args.report_steps,
                            total_div / args.report_steps, total_orth / args.report_steps, lr))
                    total_loss, total_cls_loss, total_div, total_orth  = 0.0, 0.0, 0.0, 0.0
            lr = optimizer.param_groups[0]['lr']
            # 验证（训练时先不启用拒识，只看分类性能）
            model.codebook.update_ema_during_eval = False   #验证前关闭 EMA，避免 dev 集更新 codebook
            # result = evaluate(args, devset)
            result = evaluate(args, dev_src, dev_tgt, dev_seg)
            model.codebook.update_ema_during_eval = True   # 验证完打开
            f1 = result['f1']
            with open(loss_save_path + loss_name, "a") as f:
                f.write("epoch {:3d}, valid macro f1: {:.4f}, lr: {:.4e}, batch: {:3d}\n".format(epoch, f1, lr, args.batch_size))
            if result['f1'] > best_result:
                best_result = result['f1']
                best_result_round = epoch
                args.output_model_path = os.path.join(loss_save_path, "best_model_pos_ood12.bin")
                # save_model(model, args.output_model_path)
            elif epoch - best_result_round >= args.earlystop:
                print("early stopping...")
                break
    else:
        model.load_state_dict(torch.load(best_model_path, weights_only=False), strict=False)
    # ---- 训练结束：加载 best model，构建指纹并标定阈值 ----
    model.codebook.update_ema_during_eval = False   #训练完成后关闭 EMA，避免 dev 集更新 codebook
    print("Build class fingerprints and calibrate rejection threshold...")
    if not args.no_rejection:
        # class_sequences = build_class_sequence_sets(args, model, trainset)
        class_sequences = build_class_sequence_sets(args, model, src, tgt, seg)
    else:
        class_sequences = None
    if not args.no_rejection:
        # 2) 标定每个类的序列距离阈值
        # seq_threshold, _ = calibrate_sequence_threshold(args, model, devset, class_sequences)
        seq_threshold, _ = calibrate_sequence_threshold(args, model, dev_src, dev_tgt, dev_seg, class_sequences)

        # 3) 标定每个训练序列下的重构误差阈值
        if args.use_recon_rejection:
            # seq_recon_thresholds, class_global_threshold = build_class_sequence_thresholds(
            #     args, model, devset, class_sequences,
            #     metric='hamming', pad_value=-1, max_len=L,
            #     percentile=args.recon_percentile)
            seq_recon_thresholds, class_global_threshold = build_class_sequence_thresholds(
                args, model, dev_src, dev_tgt, dev_seg, class_sequences,
                metric='hamming', pad_value=-1, max_len=L,
                percentile=args.recon_percentile)
        else:
            seq_recon_thresholds = None
    else:
        seq_threshold = L
        seq_recon_thresholds = None
        class_global_threshold = None
    # ---- 测试阶段（带拒识）----
    if args.test_path is not None and test_src.size(0) > 0:
        openset = False
        if not args.no_rejection:
            openset  = True
        model.codebook.update_ema_during_eval = False   # 最终测试前关闭
        result = evaluate(args, test_src, test_tgt, test_seg, class_sequences=class_sequences,
                    seq_threshold=seq_threshold,
                    seq_metric='hamming',
                    print_confusion_matrix=True, open_set=openset,
                    seq_recon_thresholds=seq_recon_thresholds,
                    class_global_recon_threshold=class_global_threshold)
        f1, acc, pr, rc = result['f1'], result['acc'], result['pr'], result['rc']
        with open(loss_save_path + loss_name, "a") as f:
            f.write("test macro f1: {:.4f}  acc: {:.4f} pr: {:.4f} rc: {:.4f}  best_epoch:{:4d}\n".format(
                f1, acc, pr, rc, best_result_round))
            if not args.no_rejection:
                y_pred = result['known_total'] + result['unknown_total']
                known_acc = result['known_correct'] / max(result['known_total'], 1)
                unknown_detect = result['unknown_correct_rej'] / max(result['unknown_total'], 1)
                f.write("rejected: {}/{} ({:.2f}%) | known_acc: {:.4f} | unknown_detect: {:.4f} | "
                        "false_accept: {} | false_reject: {}\n\n".format(
                            result['rejected'], y_pred,  # 注意 y_pred 是 evaluate 内部变量，需要一起返回
                            100.0 * result['rejected'] / max(y_pred, 1),
                            known_acc, unknown_detect,
                            result['false_accept'], result['false_reject']))


# if __name__ == "__main__":
#     file_list = ["cstent", "cipherspectrum_mix", "cipherspectrum_aes256", 
#                  "cipherspectrum_aes128","ustc", "cipherspectrum_cha","Edge-IIoTset"]
#     for file in file_list:
#         main(file)
if __name__ == "__main__":
    file_list = ["cstent", "cipherspectrum_mix", "cipherspectrum_aes256",
                 "cipherspectrum_aes128", "ustc", "cipherspectrum_cha", "Edge-IIoTset"]
    file_list = ['cipherspectrum_aes256', 'cipherspectrum_cha']
    file_list = ["itcnet"]
    gpu_ids = [0,1,2]          # 三张显卡的物理编号，按实际情况修改
    gpu_ids = [1]          # 三张显卡的物理编号，按实际情况修改

    if "--datasets" not in sys.argv:
        # ===== 父进程：拆分数据集，为每张卡启动一个子进程 =====
        import subprocess
        chunks = [file_list[i::len(gpu_ids)] for i in range(len(gpu_ids))]  # 轮转分配
        procs = []
        for gpu_id, chunk in zip(gpu_ids, chunks):
            if not chunk:
                continue
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)   # 子进程只能看到这一张卡
            log_path = f"train_gpu{gpu_id}.log"
            log_f = open(log_path, "w")
            p = subprocess.Popen(
                [sys.executable, "-u",os.path.abspath(__file__),
                 "--datasets", ",".join(chunk)],
                env=env, stdout=log_f, stderr=subprocess.STDOUT)
            procs.append((gpu_id, p, log_f))
            print(f"GPU {gpu_id} <- {chunk}, log: {log_path}")
        for gpu_id, p, log_f in procs:
            p.wait()
            log_f.close()
            print(f"GPU {gpu_id} finished, return code: {p.returncode}")
    else:
        # ===== 子进程：串行跑分到自己的数据集（各自独占一张卡）=====
        import argparse as _ap
        _p = _ap.ArgumentParser()
        _p.add_argument("--datasets", type=str, required=True)
        _known, _ = _p.parse_known_args()
        for file in _known.datasets.split(","):
            main(file)