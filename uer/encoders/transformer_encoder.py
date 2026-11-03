import torch
import torch.nn as nn
from uer.layers.transformer import TransformerLayer,TransformerMOELayer
from uer.layers.layer_norm import LayerNorm, T5LayerNorm
from uer.layers.relative_position_embedding import RelativePositionEmbedding
import pickle

class TransformerEncoder(nn.Module):    #基于 Transformer 的编码器类    负责将输入序列转换为有意义的特征表示
    """
    BERT encoder exploits 12 or 24 transformer layers to extract features.
    """
    def __init__(self, args):
        super(TransformerEncoder, self).__init__()
        self.mask = args.mask   # 注意力掩码类型
        self.layers_num = args.layers_num   # Transformer层数
        self.parameter_sharing = args.parameter_sharing     # 参数共享：所有层共用相同参数
        self.factorized_embedding_parameterization = args.factorized_embedding_parameterization # 因子化嵌入参数化：用于降低模型参数量
        self.layernorm_positioning = args.layernorm_positioning # LayerNorm位置：pre或post   "pre": LayerNorm在Transformer层之前 "post": LayerNorm在Transformer层之后
        self.relative_position_embedding = args.relative_position_embedding # 相对位置嵌入
        self.is_moe = args.is_moe   # 是否使用MoE(混合专家)架构
        has_bias = bool(1 - args.remove_transformer_bias)
        '''
        LayerNorm 层用于归一化，主要作用是：数值稳定性、位置作用
        Transformer 层是核心特征提取单元，包含：多头自注意力机制、前馈神经网络
        
        '''
        if self.factorized_embedding_parameterization:
            self.linear = nn.Linear(args.emb_size, args.hidden_size)
        
        if args.is_moe:     # 根据是否使用MoE选择不同的Transformer层
            transformer_layer = TransformerMOELayer
        else:
            transformer_layer = TransformerLayer

        if self.parameter_sharing:      # 根据是否参数共享构建层
            self.transformer = transformer_layer(args)
        else:
            self.transformer = nn.ModuleList(
                [transformer_layer(args) for _ in range(self.layers_num)]   #TransformerLayer(args)
            )
        if self.layernorm_positioning == "pre":     # LayerNorm类型选择
            if args.layernorm == "t5":
                self.layer_norm = T5LayerNorm(args.hidden_size)
            else:
                self.layer_norm = LayerNorm(args.hidden_size)

        if self.relative_position_embedding:    # 相对位置嵌入初始化
            self.relative_pos_emb = RelativePositionEmbedding(bidirectional=True, heads_num=args.heads_num,
                                                              num_buckets=args.relative_attention_buckets_num)


    def forward(self, emb, seg, input_ids=None ,proto=None, output_attentions=False):
        """
        Args:
            emb: [batch_size x seq_length x emb_size]
            seg: [batch_size x seq_length]
        Returns:
            hidden: [batch_size x seq_length x hidden_size]
        """
        if self.factorized_embedding_parameterization:
            emb = self.linear(emb)

        batch_size, seq_length, _ = emb.size()
        # Generate mask according to segment indicators.
        # mask: [batch_size x 1 x seq_length x seq_length]
        if self.mask == "fully_visible":
            mask = (seg > 0). \
                unsqueeze(1). \
                repeat(1, seq_length, 1). \
                unsqueeze(1)
            mask = mask.float()
            mask = (1.0 - mask) * -10000.0
        elif self.mask == "causal":
            mask = torch.ones(seq_length, seq_length, device=emb.device)
            mask = torch.tril(mask)
            mask = (1.0 - mask) * -10000
            mask = mask.repeat(batch_size, 1, 1, 1)
        else:
            mask_a = (seg == 1). \
                unsqueeze(1). \
                repeat(1, seq_length, 1). \
                unsqueeze(1).float()

            mask_b = (seg > 0). \
                unsqueeze(1). \
                repeat(1, seq_length, 1). \
                unsqueeze(1).float()

            mask_tril = torch.ones(seq_length, seq_length, device=emb.device)
            mask_tril = torch.tril(mask_tril)
            mask_tril = mask_tril.repeat(batch_size, 1, 1, 1)

            mask = (mask_a + mask_b + mask_tril >= 2).float()
            mask = (1.0 - mask) * -10000.0

        hidden = emb

        if self.relative_position_embedding:
            position_bias = self.relative_pos_emb(hidden, hidden)
        else:
            position_bias = None
        
        probss = []
        gate_loss = 0.0
        for i in range(self.layers_num):
            if self.parameter_sharing:
                if self.is_moe:
                    hidden,banlance_loss = self.transformer(hidden, mask, position_bias=position_bias, expert_input_ids=input_ids ,proto=proto)
                    gate_loss+=banlance_loss
                else:
                    hidden,probs = self.transformer(hidden, mask, position_bias=position_bias)
            else:
                if self.is_moe:
                    hidden,banlance_loss = self.transformer[i](hidden, mask, position_bias=position_bias, expert_input_ids=input_ids, proto=proto)
                    gate_loss+=banlance_loss
                else:
                    hidden,probs = self.transformer[i](hidden, mask, position_bias=position_bias)
            probss.append(probs)

        # with open("/mnt/data/zgm/ET-BERT/fine-tuning/attentions/attention.pkl","wb") as f:
        #     pickle.dump(probss,f)

        if self.layernorm_positioning == "pre":     #默认为post
            if self.is_moe:
                return self.layer_norm(hidden),gate_loss
            else:
                return self.layer_norm(hidden)
        else:
            if self.is_moe:
                return hidden,gate_loss
            else:
                return hidden if not output_attentions else hidden, probss  # 其中 hidden[:, 0, :] 就是CLS向量
