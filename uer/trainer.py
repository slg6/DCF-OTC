import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from uer.model_loader import load_model
from uer.model_saver import save_model
from uer.model_builder import build_model
from uer.utils.optimizers import *
from uer.utils import *
from uer.utils.vocab import Vocab
from uer.utils.seed import set_seed
from tqdm import tqdm
from uer.config import K
import torch.nn.functional as F

def load_or_initialize_parameters(args, model):
    try:
        # 尝试加载预训练模型的权重
        pretrained_weights = torch.load(args.pretrained_model_path)
        model_dict = model.state_dict() #你可以直接修改 state_dict 中的张量，然后使用 load_state_dict() 将修改后的权重加载到模型中。
        # 过滤掉不匹配的权重
        filtered_weights = {k: v for k, v in pretrained_weights.items() if k in model_dict and v.shape == model_dict[k].shape}
        # 更新模型的权重
        model_dict.update(filtered_weights)
        model.load_state_dict(model_dict)
        print("Successfully loaded pre-trained model weights.")
    except Exception as e:
        print(f"Failed to load pre-trained model weights: {e}")
        print("Initializing model parameters randomly.")
        # 如果加载失败，可以随机初始化模型参数
        model.apply(model._init_weights)
def train_and_validate(args):
    set_seed(args.seed)

    # Load vocabulary.
    print("Load vocabulary.")

    args.tokenizer = str2tokenizer[args.tokenizer](args)    #BertTokenizer(args)
    args.vocab = args.tokenizer.vocab
    #slg
    # train_loader = str2dataloader[args.target](args, args.dataset_path, args.batch_size, 0, 1, True)
    # train_loader = iter(train_loader)
    # flag = 0
    # for i in next(train_loader):
    #     flag += 1
    # print(flag)
    #slg
    # Build model.
    print("Build model.")
    model = build_model(args)   #创建模型结构
    # for name,parameters in model.named_parameters():
    #     print(name,':',parameters.size())
    # Load or initialize parameters.
    # if args.pretrained_model_path is not None:
    #     # Initialize with pretrained model.
    #     # model = load_model(model, args.pretrained_model_path) #加载预训练模型
    #     load_or_initialize_parameters(args, model)
    # else:
    # Initialize with normal distribution.
    for n, p in list(model.named_parameters()):
        if "gamma" not in n and "beta" not in n and "codebook" not in n:
            p.data.normal_(0, 0.02)

    if args.dist_train:
        # Multiprocessing distributed mode.
        print("Multiprocessing distributed mode.")
        mp.spawn(worker, nprocs=args.ranks_num, args=(args.gpu_ranks, args, model), daemon=False)
    elif args.single_gpu:
        # Single GPU mode.
        print("Single GPU mode.")
        worker(args.gpu_id, None, args, model)
    else:
        # CPU mode.
        print("CPU mode.")
        worker(None, None, args, model)


class Trainer(object):
    def __init__(self, args):
        self.current_step = 1
        self.total_steps = args.total_steps
        self.accumulation_steps = args.accumulation_steps       #accumulation_steps 参数用于实现梯度累积，通过在多个小批次上累积梯度，模拟大批次训练的效果。这种方法可以在不增加硬件负担的情况下提高模型的训练效果，尤其适用于内存受限的场景。
        self.report_steps = args.report_steps
        self.save_checkpoint_steps = args.save_checkpoint_steps

        self.output_model_path = args.output_model_path

        self.start_time = time.time()
        self.total_loss = 0.0
        
        self.dist_train = args.dist_train
        self.batch_size = args.batch_size
        self.world_size = args.world_size

    def forward_propagation(self, batch, model):

        raise NotImplementedError

    def report_and_reset_stats(self):

        raise NotImplementedError

    def train(self, args, gpu_id, rank, loader, model, optimizer, scheduler):
        model.train()
        # model.codebook.weight.requires_grad = False
        loader_iter = iter(loader)
        
        while True:
            if self.current_step == self.total_steps + 1:       #每一步仅训练一个batch_size，不是所有的数据
                break
            batch = list(next(loader_iter))     #先加载22560个实例，一次从中取出一个batch的实例，取完以后，再补充。 一个batch实例含有batch_size个样本
            self.seq_length = batch[0].size(1)
            if gpu_id is not None:
                for i in range(len(batch)):
                    batch[i] = batch[i].cuda(gpu_id)

            loss = self.forward_propagation(batch, model)

            # if args.fp16:
            #     with args.amp.scale_loss(loss, optimizer) as scaled_loss:
            #         scaled_loss.backward()
            # else:
            loss.backward()

            if self.current_step % self.accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
            # print(f"self.current_step % self.report_steps, {self.current_step} % {self.report_steps}")
            if self.current_step % self.report_steps == 0 and \
                    (not self.dist_train or (self.dist_train and rank == 0)):
                
                self.report_and_reset_stats()
                self.start_time = time.time()

            if self.current_step % self.save_checkpoint_steps == 0 and \
                    (not self.dist_train or (self.dist_train and rank == 0)):
                save_model(model, self.output_model_path + "-" + str(self.current_step))    #保存中间模型

            self.current_step += 1


class MlmTrainer(Trainer):
    def __init__(self, args):
        super(MlmTrainer, self).__init__(args)
        self.total_correct = 0.0
        self.total_denominator = 0.0

    def forward_propagation(self, batch, model):
        src, tgt, seg = batch
        loss_info = model(src, tgt, seg)
        loss, correct, denominator = loss_info
        self.total_loss += loss.item()
        self.total_correct += correct.item()
        self.total_denominator += denominator.item()
        loss = loss / self.accumulation_steps
        return loss

    def report_and_reset_stats(self):
        done_tokens = self.batch_size * self.seq_length * self.report_steps
        if self.dist_train:
            done_tokens *= self.world_size
        print("| {:8d}/{:8d} steps"
              "| {:8.2f} tokens/s"
              "| loss {:7.2f}"
              "| acc: {:3.3f}".format(
            self.current_step,
            self.total_steps,
            done_tokens / (time.time() - self.start_time),
            self.total_loss / self.report_steps,
            self.total_correct / self.total_denominator))

        self.total_loss = 0.0
        self.total_correct = 0.0
        self.total_denominator = 0.0
        
class BertPacketsTrainer(Trainer):
    def __init__(self, args):
        super(BertPacketsTrainer, self).__init__(args)
        self.total_loss_sp = 0.0
        self.total_correct_sp = 0.0
        self.total_instances = 0.0

        # self.total_loss += loss.item()
        # self.total_loss_mlm += loss_mlm.item()
        # self.total_loss_sp += loss_sp.item()
        # self.total_loss_fm += loss_fm.item()
        # self.total_correct_mlm += correct_mlm.item()
        # self.total_correct_sp += correct_sp.item()
        # self.total_correct_fm += correct_fm.item()
        # self.total_denominator += denominator_mlm.item()
        # self.total_denominator_fm += denominator_fm.item()
        # self.total_instances += src.size(0)     #总实例，与batch_size一样
        self.total_loss_vq = 0.0
        self.total_loss_vq_log = 0.0
        self.total_loss_div = 0.0
        self.total_h_std = 0.0
        self.global_hit_count = torch.zeros(K)
        self.global_hit_count_ = torch.zeros(K)
        self.total_loss_packet_contrastive = 0.0
        # self.register_buffer('global_hit_count', torch.zeros(K))  # K=256，和 model.py 里保持一致


        self.total_loss_mlm = 0.0
        self.total_correct_mlm = 0.0
        self.total_loss_fm = 0.0
        self.total_correct_fm = 0.0
        self.total_denominator = 0.0
        self.total_denominator_fm = 0.0
        self.load_balance_alpha = args.moebert_load_balance
        self.is_moe = args.is_moe
        self.log_file = args.log_file

    def forward_propagation(self, batch, model):

        src, tgt_mlm, label, field_modified, seg, proto = batch
        loss_info, indices, h, z_q, pack_mask = model(src, (tgt_mlm, label, field_modified), seg,proto)

        loss_mlm, loss_sp, loss_fm,correct_mlm, correct_sp, correct_fm, denominator_mlm, denominator_fm = loss_info
        # gate_loss = 0.0
        loss_pre = loss_mlm/10 + loss_sp + loss_fm/10 #+ self.load_balance_alpha * gate_loss  #MLM损失   序列损失  包位置损失  MOE下的平衡 （此处的权重是不是能修改一下）
        
        #包对比学习
        B, L, d = h.shape
        valid_mask = pack_mask.bool()  # (B, 20)
        # 1. 拉平有效包
        h_valid = h[valid_mask]  # (N_valid, d) 191=valid_mask
        h_valid = F.normalize(h_valid, dim=-1)
        # 2. 给每个有效包标记它属于哪个 flow
        flow_ids = torch.arange(B, device=h.device).view(B, 1).expand(-1, L)  # (B, 20) 0-23
        flow_ids_valid = flow_ids[valid_mask]  # (N_valid,) 191=valid_mask
        # 3. 计算所有有效包两两相似度
        temperature = 0.1
        sim = torch.matmul(h_valid, h_valid.t()) / temperature  # (N, N)
        # 4. 正样本 mask：同 flow，且不是自己
        N = h_valid.size(0)
        pos_mask = (flow_ids_valid.unsqueeze(1) == flow_ids_valid.unsqueeze(0))  # (N, N)
        pos_mask.fill_diagonal_(False)
        # 5. InfoNCE
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim_stable = sim - sim_max.detach()
        exp_sim = torch.exp(sim_stable)
        pos_sum = (exp_sim * pos_mask.float()).sum(dim=1)  # (N,)
        all_sum = exp_sim.sum(dim=1)  # (N,)
        loss_packet_contrastive = -torch.log((pos_sum + 1e-8) / (all_sum + 1e-8))
        loss_packet_contrastive = loss_packet_contrastive.mean()
        
        #VQ损失  码本
        e_k = model.codebook(indices)
        # loss_vq = torch.mean((h.detach() - e_k)**2) + 0.25 * torch.mean((h - e_k.detach()) ** 2)
        diff_sq = (h.detach() - e_k) ** 2
        loss_vq_log = (diff_sq.mean(dim=-1) * pack_mask).sum() / (pack_mask.sum() + 1e-6)
        diff_sq2 = (h - e_k.detach()) ** 2
        COMMITMENT_COEF = 1  # 可调
        loss_vq = COMMITMENT_COEF * (diff_sq2.mean(dim=-1) * pack_mask).sum() / (pack_mask.sum() + 1e-6)

        # p = torch.bincount(indices.flatten(), minlength=model.K) / len(indices.flatten())   #统计基元使用频率，计算负熵，
        valid_indices = indices[pack_mask.bool()]      # 展平后只保留有效包，这一步得到的就是一维的张量
        if valid_indices.numel() > 0:
            #torch.bincount计算一维非负整数张量中，每个整数从 0 到 max (input) 出现的次数。
            #输入必须是一维非负整数类型
            p = torch.bincount(valid_indices.flatten(), minlength=model.K) / valid_indices.numel()
            loss_div = (p * torch.log(p + 1e-10)).sum()     #不可导
        else:
            loss_div = torch.tensor(0.0, device=indices.device)
        
        # loss_div = (p * torch.log(p + 1e-10)).sum() #并通过最小化负熵（即最大化熵）来保证码本利用的均衡性。
        WARMUP_STEPS = 3000     #前三千步优化码本为主
        if self.current_step < WARMUP_STEPS:
            loss = loss_pre + loss_vq + 0.05 * loss_packet_contrastive # 码本h主导期
        else:
            loss = loss_pre + 0.5 * loss_vq + 0.05 * loss_packet_contrastive         # 正常联合训练
        h_std = h.std(dim=0).mean()

        # 超参数
        gamma_base = 0.9          # 最小衰减系数
        alpha = 1.0               # 频率敏感度
        dead_threshold = 0.0001     # 死亡基元 = 那些 EMA 使用次数 < 平均使用次数 × 0.0001 的基元。
        #EMA更新码本
        with torch.no_grad():
            #这本质上就是在计算一个动态的质心（centroid），让码本向量始终位于它所吸引的那些编码器输出的中心。
            # 从 pack_mask 中取出有效位置（pack_mask 是你在 forward 中生成的 (B, 20) mask）
            valid_mask = pack_mask.bool()                    # (B, 20)
            valid_indices = indices[valid_mask]              # (N_valid,)
            h_valid = h[valid_mask]                          # (N_valid, d_code)
            # if valid_indices.numel() > 0:
            # one_hot 后 shape 为 (N_valid, K)
            encodings = torch.nn.functional.one_hot(valid_indices, model.K).float()

            # 1. 统计当前 batch 有效位置中，每个基元被激活的总次数
            # 对 N_valid 维度求和 → (K,)
            batch_cluster_size = encodings.sum(dim=0)   #batch_cluster_size 本质上是当前 batch 中每个基元被激活的总次数。
            self.global_hit_count += batch_cluster_size.cpu()  # <-- 新增：跨 batch 累积
            self.global_hit_count_ += batch_cluster_size.cpu()  # <-- 新增：跨 batch 累积

            # 2. 统计每个基元所吸引的全部 h 向量的和
            # encodings: (N_valid, K), h_valid: (N_valid, d_code) → (K, d_code)
            sum_h = torch.einsum('nk,nd->kd', encodings, h_valid)

            avg_size = model.ema_cluster_size.mean() + 1e-8
      
            # 放在 EMA 更新块的开头
            ema_decay = min(0.99, 0.9 + 0.00003 * self.current_step)  # 从 0.9 慢慢涨到 0.99
            model.ema_cluster_size = ema_decay * model.ema_cluster_size + (1 - ema_decay) * batch_cluster_size
            model.ema_embed = ema_decay * model.ema_embed + (1 - ema_decay) * sum_h
            
            #PyTorch 自动将形状为[1024, 1]的张量，在第二个维度上复制 256 次，扩展成[1024, 256]的形状，然后与sum_h进行逐元素相乘。
            # --- 在线死基元重置 ---
            # 判定条件：平滑使用次数低于平均的 dead_threshold 倍
            dead_mask = model.ema_cluster_size < avg_size * dead_threshold   # (K,) bool
            reset_interval = 100 if self.current_step < 5000 else 1000
            if dead_mask.any() and self.current_step % reset_interval == 0:
            # if dead_mask.any() and self.current_step % 1000 == 0:  # 每1000步检查一次死基元，避免过于频繁地重置
                # 从当前 batch 中随机选取等量的 h 向量作为新基元
                num_dead = dead_mask.sum().item()
                valid_mask_flat = pack_mask.bool().reshape(-1)  # (B*20,)
                
                h_flat = h.reshape(-1, model.d_code)   
                h_valid = h_flat[valid_mask_flat]                    # (total_N, d_code)
                
                rand_idx = torch.randint(0, h_valid.shape[0], (num_dead,), device=indices.device)
                new_vectors = h_valid[rand_idx].clone()
                
                # 替换死基元向量
                INITIAL_RESET_SIZE = 20.0  # 相当于被“伪激活”了20次，给一个"免死金牌"，让它能活更久
                model.ema_cluster_size[dead_mask] = INITIAL_RESET_SIZE
                model.ema_embed[dead_mask] = new_vectors * INITIAL_RESET_SIZE

            # --- 常规的码本向量更新（使用 EMA 累积量计算质心） ---
            new_codebook = model.ema_embed / (model.ema_cluster_size.unsqueeze(1) + 1e-5)
            model.codebook.weight.data.copy_(new_codebook)


        # loss = loss_mlm  #onlymfm
        self.total_loss += loss.item()
        self.total_loss_mlm += loss_mlm.item()
        self.total_loss_sp += loss_sp.item()
        self.total_loss_fm += loss_fm.item()
        self.total_loss_vq += loss_vq.item()
        self.total_loss_packet_contrastive += loss_packet_contrastive.item()
        self.total_loss_vq_log += loss_vq_log.item()
        self.total_h_std += h_std.item()
        self.total_loss_div += loss_div.item()
        self.total_correct_mlm += correct_mlm.item()
        self.total_correct_sp += correct_sp.item()
        self.total_correct_fm += correct_fm.item()
        self.total_denominator += denominator_mlm.item()
        self.total_denominator_fm += denominator_fm.item()
        self.total_instances += src.size(0)     #总实例，与batch_size一样
        loss = loss / self.accumulation_steps       #1

        return loss
    def report_and_reset_stats(self):
        import logging
        import time
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                            handlers=[
                                logging.FileHandler(self.log_file), #表示log写入文件
                                logging.StreamHandler()     #表示log打印终端
                            ])
        logger = logging.getLogger(__name__)
        done_tokens = self.batch_size * self.seq_length * self.report_steps
        if self.dist_train:
            done_tokens *= self.world_size

        global_active = (self.global_hit_count > 0).sum().item()
        global_active_ = (self.global_hit_count_ > 0).sum().item()
        p_global = self.global_hit_count / (self.global_hit_count.sum() + 1e-10)
        global_entropy = -(p_global * torch.log(p_global + 1e-10)).sum().item()
        
        log_message = (
        f"| {self.current_step:8d}/{self.total_steps:8d} steps"
        f"| {time.time() - self.start_time:3.3f} s"
        f"| {done_tokens / (time.time() - self.start_time):8.2f} tokens/s"
        f"| loss {self.total_loss / self.report_steps:7.2f}"
        f"| loss_mlm: {self.total_loss_mlm / self.report_steps:3.3f}"
        f"| loss_sp: {self.total_loss_sp / self.report_steps:3.3f}"
        f"| loss_fm: {self.total_loss_fm / self.report_steps:3.3f}"
        f"| loss_vq: {self.total_loss_vq / self.report_steps:3.3f}"
        f"| loss_vq_log: {self.total_loss_vq_log / self.report_steps:3.3f}"
        f"| h_std: {self.total_h_std / self.report_steps:3.3f}"
        f"| loss_packet_contrastive: {self.total_loss_packet_contrastive / self.report_steps:3.3f}"
        f"| global_active: {global_active:3d}"
        f"| global_active_100: {global_active_:3d}"
        f"| global_entropy: {global_entropy:3.3f}"
        f"| loss_div: {self.total_loss_div / self.report_steps:3.3f}"
        f"| acc_mlm: {self.total_correct_mlm / self.total_denominator:3.3f}"
        f"| acc_sp: {self.total_correct_sp / self.total_instances:3.3f}"
        f"| acc_fm: {self.total_correct_fm / self.total_denominator_fm:3.3f}"

    )
        if self.current_step % 100 == 0:
            self.global_hit_count_ = torch.zeros(K) 
        logger.info(log_message)

        self.total_loss_vq = 0.0
        self.total_loss_vq_log = 0.0
        self.total_loss_div = 0.0
        self.total_h_std = 0.0
        self.total_loss_packet_contrastive = 0.0

        self.total_loss, self.total_loss_mlm, self.total_loss_sp, self.total_loss_fm = 0.0, 0.0, 0.0, 0.0
        self.total_correct_mlm, self.total_denominator, self.total_denominator_fm = 0.0, 0.0, 0.0
        self.total_correct_sp, self.total_correct_fm, self.total_instances = 0.0, 0.0, 0.0

class BertTrainer(Trainer):
    def __init__(self, args):
        super(BertTrainer, self).__init__(args)
        self.total_loss_sp = 0.0
        self.total_correct_sp = 0.0
        self.total_instances = 0.0

        self.total_loss_mlm = 0.0
        self.total_correct_mlm = 0.0
        self.total_denominator = 0.0
        self.load_balance_alpha = args.moebert_load_balance
        self.is_moe = args.is_moe

    def forward_propagation(self, batch, model):
        debug_mode = False
        if debug_mode:
            print("In function forward_propagation(self, batch, model):")
            print("type of batch:",type(batch))
            print("type of the content of batch:",[type(elem) for elem in batch])
        if len(batch)==5:
            src, tgt_mlm, tgt_sp, seg, proto = batch
            '''
            batch具体实例：
            instance = (
                [101, 7592, 103, 102, 2023, 103, 1034, 102, 0, 0, ...],  # src 输入序列（Token ID 列表）。
                [(3, 2129), (6, 2003)],                              #tgt_mlm 3，6被掩码  掩码前值为2129，2003
                0,                                                    #tgt_sp  0 表示句子对是连续的。  1 表示句子对是随机拼接的。
                [4, 8],                                              #seg  Segment A 结束于索引 4，Segment B 结束于索引 8
                0,                                                     #A包
                0                                                     #proto 协议为TCP
            )

            '''
            loss_info = model(src, (tgt_mlm, tgt_sp), seg, proto)  #loss_info是model的forward函数计算出来的  
            #传给target层的是（含有CLS的张量，tgt）   tgt= (tgt_mlm, tgt_sp)   tgt_mlm是用于字词预测的不用修改
            #修改tgt_sp，代表包的位置即可。    只需要修改数据的传入方式即可
            '''
            model
            Pretraining models consist of three parts:
                - embedding
                - encoder
                - target
            ''' 
        else:
            src, tgt_mlm, tgt_sp, seg = batch
            loss_info = model(src, (tgt_mlm, tgt_sp), seg)
        
        if self.is_moe:
            loss_mlm, loss_sp, correct_mlm, correct_sp, denominator, gate_loss = loss_info
        else:
            loss_mlm, loss_sp, correct_mlm, correct_sp, denominator = loss_info
            gate_loss = 0.0
        loss = loss_mlm/10 + loss_sp + self.load_balance_alpha * gate_loss  #MLM损失   序列损失   MOE下的平衡
        self.total_loss += loss.item()
        self.total_loss_mlm += loss_mlm.item()
        self.total_loss_sp += loss_sp.item()
        self.total_correct_mlm += correct_mlm.item()
        self.total_correct_sp += correct_sp.item()
        self.total_denominator += denominator.item()
        self.total_instances += src.size(0)
        loss = loss / self.accumulation_steps

        return loss

    def report_and_reset_stats(self):
        done_tokens = self.batch_size * self.seq_length * self.report_steps
        if self.dist_train:
            done_tokens *= self.world_size

        print("| {:8d}/{:8d} steps"
              "| {:3.3f} s"
              "| {:8.2f} tokens/s"
              "| loss {:7.2f}"
              "| loss_mlm: {:3.3f}"
              "| loss_sp: {:3.3f}"
              "| acc_mlm: {:3.3f}"
              "| acc_sp: {:3.3f}".format(
            self.current_step,
            self.total_steps,
            (time.time() - self.start_time),
            done_tokens / (time.time() - self.start_time),
            self.total_loss / self.report_steps,
            self.total_loss_mlm / self.report_steps,
            self.total_loss_sp / self.report_steps,
            self.total_correct_mlm / self.total_denominator,
            self.total_correct_sp / self.total_instances))

        self.total_loss, self.total_loss_mlm, self.total_loss_sp = 0.0, 0.0, 0.0
        self.total_correct_mlm, self.total_denominator = 0.0, 0.0
        self.total_correct_sp, self.total_instances = 0.0, 0.0


class AlbertTrainer(BertTrainer):
    pass


class LmTrainer(MlmTrainer):
    pass


class BilmTrainer(Trainer):
    def __init__(self, args):
        super(BilmTrainer, self).__init__(args)
        self.total_loss_forward, self.total_loss_backward = 0.0, 0.0
        self.total_correct_forward, self.total_correct_backward = 0.0, 0.0
        self.total_denominator = 0.0

    def forward_propagation(self, batch, model):
        src, tgt_forward, tgt_backward, seg = batch
        loss_info = model(src, (tgt_forward, tgt_backward), seg)
        loss_forward, loss_backward, correct_forward, correct_backward, denominator = loss_info
        loss = loss_forward + loss_backward
        self.total_loss += loss.item()
        self.total_loss_forward += loss_forward.item()
        self.total_loss_backward += loss_backward.item()
        self.total_correct_forward += correct_forward.item()
        self.total_correct_backward += correct_backward.item()
        self.total_denominator += denominator.item()
        loss = loss / self.accumulation_steps
        return loss

    def report_and_reset_stats(self):
        done_tokens = self.batch_size * self.seq_length * self.report_steps
        if self.dist_train:
            done_tokens *= self.world_size
        print("| {:8d}/{:8d} steps"
                  "| {:8.2f} tokens/s"
                  "| loss {:7.2f}"
                  "| loss_forward {:3.3f}"
                  "| loss_backward {:3.3f}"
                  "| acc_forward: {:3.3f}"
                  "| acc_backward: {:3.3f}".format(
                    self.current_step,
                    self.total_steps, 
                    done_tokens / (time.time() - self.start_time), 
                    self.total_loss / self.report_steps,
                    self.total_loss_forward / self.report_steps,
                    self.total_loss_backward / self.report_steps,
                    self.total_correct_forward / self.total_denominator,
                    self.total_correct_backward / self.total_denominator))

        self.total_loss, self.total_loss_forward, self.total_loss_backward = 0.0, 0.0, 0.0
        self.total_correct_forward, self.total_correct_backward, self.total_denominator = 0.0, 0.0, 0.0


class ClsTrainer(Trainer):
    def __init__(self, args):
        super(ClsTrainer, self).__init__(args)
        self.total_correct = 0.0
        self.total_instances = 0.0

    def forward_propagation(self, batch, model):
        src, tgt, seg = batch
        loss_info = model(src, tgt, seg)
        loss, correct = loss_info
        self.total_loss += loss.item()
        self.total_correct += correct.item()
        self.total_instances += src.size(0)
        loss = loss / self.accumulation_steps
        return loss

    def report_and_reset_stats(self):
        done_tokens = self.batch_size * self.seq_length * self.report_steps
        if self.dist_train:
            done_tokens *= self.world_size
        print("| {:8d}/{:8d} steps"
              "| {:8.2f} tokens/s"
              "| loss {:7.2f}"
              "| acc: {:3.3f}".format(
            self.current_step,
            self.total_steps,
            done_tokens / (time.time() - self.start_time),
            self.total_loss / self.report_steps,
            self.total_correct / self.total_instances))

        self.total_loss = 0.0
        self.total_correct = 0.0
        self.total_instances = 0.0


class Seq2seqTrainer(Trainer):
    def __init__(self, args):
        super(Seq2seqTrainer, self).__init__(args)
        self.total_correct = 0.0
        self.total_denominator = 0.0

    def forward_propagation(self, batch, model):
        src, tgt_in, tgt_out, seg = batch
        loss_info = model(src, (tgt_in, tgt_out, src), seg)
        loss, correct, denominator = loss_info
        self.total_loss += loss.item()
        self.total_correct += correct.item()
        self.total_denominator += denominator.item()

        loss = loss / self.accumulation_steps

        return loss

    def report_and_reset_stats(self):
        done_tokens = self.batch_size * self.seq_length * self.report_steps
        if self.dist_train:
            done_tokens *= self.world_size

        print("| {:8d}/{:8d} steps"
              "| {:8.2f} tokens/s"
              "| loss {:7.2f}"
              "| acc: {:3.3f}".format(
            self.current_step,
            self.total_steps,
            done_tokens / (time.time() - self.start_time),
            self.total_loss / self.report_steps,
            self.total_correct / self.total_denominator))

        self.total_loss = 0.0
        self.total_correct = 0.0
        self.total_denominator = 0.0


class T5Trainer(Seq2seqTrainer):
    pass


class PrefixlmTrainer(MlmTrainer):
    pass


str2trainer = {"bert": BertTrainer, "bertflow": BertTrainer,"bertpackets":BertPacketsTrainer,"mlm": MlmTrainer, "lm": LmTrainer,
               "albert": AlbertTrainer, "bilm": BilmTrainer, "cls": ClsTrainer,
               "seq2seq": Seq2seqTrainer, "t5": T5Trainer}

def worker(proc_id, gpu_ranks, args, model):    
    """
    Args:
        proc_id: The id of GPU for single GPU mode;
                 The id of process (and GPU) for multiprocessing distributed mode.
        gpu_ranks: List of ranks of each process.
    """
    set_seed(args.seed)
    
    print("Starting worker function")

    if args.dist_train:
        rank = gpu_ranks[proc_id]
        gpu_id = proc_id
    elif args.single_gpu:
        rank = None
        gpu_id = proc_id
    else:
        rank = None
        gpu_id = None
    print("train loader constructing...")
    if args.dist_train:     #  if args.world_size > 1:   args.dist_train = True
        train_loader = str2dataloader[args.target](args, args.dataset_path, args.batch_size, rank, args.world_size, True)
    else:
        train_loader = str2dataloader[args.target](args, args.dataset_path, args.batch_size, 0, 1, True)
    '''
    train_loader具体实例：
    instance = (            
        [101, 7592, 103, 102, 2023, 103, 1034, 102, 0, 0, ...],  # src
        [(3, 2129), (6, 2003)],                              # 3，6被掩码  掩码前值为2129，2003
        0,                                                    #  0 表示句子对是连续的。  1 表示句子对是随机拼接的。
        [4, 8],                                              # Segment A 结束于索引 4，Segment B 结束于索引 8
        0                                                     # 协议为TCP
    )

    train_loader修改目标：
    instance = (            
        [101, 7592, 103, 102, 2023, 103, 1034, 102, 0, 0, ...],  # src  一个数据包的内容
        [(3, 2129), (6, 2003)],                              # 3，6被掩码  掩码前值为2129，2003  不变
        0,                                                    #  0 表示句子对是连续的。  1 表示句子对是随机拼接的。0,1,2,3,4
        [4, 8],                                              #  token_A 结束于索引 4，tokne_B 结束于索引 8  
        0,                                                    #  0 表示A包在数据流中的位置
        0                                                     # 协议为TCP
    )

    '''
    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)
        model.cuda(gpu_id)

    # Build optimizer.
    print("build optomizer...")
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "gamma", "beta"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay_rate": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay_rate": 0.0}
    ]
    if args.optimizer in ["adamw"]: #default
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    else:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate,
                                                  scale_parameter=False, relative_step=False)
    if args.scheduler in ["constant"]:
        scheduler = str2scheduler[args.scheduler](optimizer)
    elif args.scheduler in ["constant_with_warmup"]:
        scheduler = str2scheduler[args.scheduler](optimizer, args.total_steps*args.warmup)
    else:   #default
        scheduler = str2scheduler[args.scheduler](optimizer, args.total_steps*args.warmup, args.total_steps)

    # if args.fp16:   #false
    #     try:
    #         from apex import amp
    #     except ImportError:
    #         raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    #     model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    #     args.amp = amp

    if args.dist_train:
        print("Initialize multiprocessing distributed training environment...")
        # Initialize multiprocessing distributed training environment.
        dist.init_process_group(backend=args.backend,
                                init_method=args.master_ip,
                                world_size=args.world_size,
                                rank=rank)
        # model = DistributedDataParallel(model, device_ids=[gpu_id], find_unused_parameters=True)
        model = DistributedDataParallel(model, device_ids=[gpu_id], find_unused_parameters=False)
        print("Worker %d is training ... " % rank)
    else:
        print("Worker is training ...")

    trainer = str2trainer[args.target](args)    #'bertpackets' BertTrainer(args)
    trainer.train(args, gpu_id, rank, train_loader, model, optimizer, scheduler)
