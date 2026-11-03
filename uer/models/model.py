import torch
import torch.nn as nn
from uer.config import K, d_code, vq_after_layer

class Model(nn.Module):
    """
    Pretraining models consist of three parts:
        - embedding
        - encoder
        - target
    """
    def __init__(self, args, embedding, encoder, target):
        super(Model, self).__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.target = target
        self.is_moe = args.is_moe       #false


        self.codebook = nn.Embedding(K, d_code)
        self.K = K
        self.d_code = d_code
        self.proj_in = nn.Linear(768, d_code)
        self.proj_out = nn.Linear(d_code, 768)
        # 码本初始化
        nn.init.uniform_(self.codebook.weight, -1.0/K, 1.0/K)
        self.register_buffer('ema_cluster_size', torch.zeros(K))
        # self.register_buffer('ema_embed', torch.zeros(K, d_code))  # 修正：零初始化
        self.register_buffer('ema_embed', self.codebook.weight.data.clone())
        #从诞生起就冻结码本，仅通过 EMA 更新
        self.codebook.weight.requires_grad = False
        # 新增：指定切分层（例如 12 层中第 6 层后切分）
        self.vq_after_layer = vq_after_layer  # 例如 6

    def quantize(self, h):
        # h: (B, L_pack, d_code)
        
        #计算每个 h_i 与码本中所有 K 个基元的欧氏距离。
        dist = torch.cdist(h, self.codebook.weight)
        indices = torch.argmin(dist, dim=-1)	#查找最近邻索引
        z_q = self.codebook(indices)	#查表得到量化向量
        # 直通估计 STE
        z_q = h + (z_q - h).detach()
        return z_q, indices
        

    def forward(self, src, tgt, seg, proto=None):
        emb = self.embedding(src, seg)
        
        hidden = emb
        batch_size, seq_length, _ = hidden.size()
        
        mask = (seg > 0).unsqueeze(1).repeat(1, seq_length, 1).unsqueeze(1).float()
        mask = (1.0 - mask) * -10000.0

        position_bias = None
        
        # ============================================
        # Lower Encoder：前 vq_after_layer 层
        # ============================================
        # 注意：默认 parameter_sharing=False，self.encoder.transformer 是 nn.ModuleList
        for i in range(self.vq_after_layer):
            hidden, _ = self.encoder.transformer[i](hidden, mask, position_bias=position_bias)
        
        # ============================================
        # VQ 处理（在 Lower 和 Upper 之间插入）
        # ============================================
        #你每步实际量化的只有 batch_size * 20 个向量（默认 batch 24 时仅 480 个），
        # 但码本大小 K=256。
        # 如果这 480 个向量本身多样性就不足，
        # 256 个基元会在 EMA 下抢夺同一簇数据，最终全部挤在一起。
        pack_list = [i * 11 for i in range(1, 21)]
        pack_hidden = hidden[:, pack_list, :]          # (B, 20, 768)
        pack_mask = (seg[:, pack_list] != 0).float()   # (B, 20)
        
        h = self.proj_in(pack_hidden)                  # (B, 20, 256)
        h = nn.functional.layer_norm(h, h.shape[-1:])
        z_q, indices = self.quantize(h)
        hidden[:, pack_list, :] = self.proj_out(z_q)   # 替换回 hidden
        
        # ============================================
        # Upper Encoder：剩余层
        # ============================================
        for i in range(self.vq_after_layer, self.encoder.layers_num):
            hidden, _ = self.encoder.transformer[i](hidden, mask, position_bias=position_bias)
        
        
        output = hidden
        
        # ============================================
        # 任务头
        # ============================================
        loss_info = self.target(output, tgt)
        return loss_info, indices, h, z_q, pack_mask


def from_pretrained(self,args, model):
    try:
        # 尝试加载预训练模型的权重
        pretrained_weights = torch.load(args.pretrained_model_path)
        model_dict = self.state_dict() #你可以直接修改 state_dict 中的张量，然后使用 load_state_dict() 将修改后的权重加载到模型中。
        # 过滤掉不匹配的权重
        filtered_weights = {k: v for k, v in pretrained_weights.items() if k in model_dict and v.shape == model_dict[k].shape}
        # 更新模型的权重
        model_dict.update(filtered_weights)
        self.load_state_dict(model_dict)
        print("Successfully loaded pre-trained model weights.")
    except Exception as e:
        print(f"Failed to load pre-trained model weights: {e}")
        print("Initializing model parameters randomly.")
        # 如果加载失败，可以随机初始化模型参数
        self.apply(model._init_weights)