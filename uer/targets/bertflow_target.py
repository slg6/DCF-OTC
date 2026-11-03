import torch
import torch.nn as nn
from uer.targets import *


class BertFlowTarget(MlmTarget):
    """
    BERT流量分类目标层:
    1. MLM(掩码语言模型): 预测被掩码的token
    2. MSP(混合句子预测): 预测流量序列的类别
    """

    def __init__(self, args, vocab_size):
        super(BertFlowTarget, self).__init__(args, vocab_size)
        # MSP.
        self.msp_linear_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.msp_linear_2 = nn.Linear(args.hidden_size, 5)
    def forward(self, memory_bank, tgt):
        """
        前向传播计算损失
        
        参数:
            memory_bank: 编码器输出 [batch_size x seq_length x hidden_size]
            tgt: 包含两个目标的元组:
                - tgt_mlm: MLM目标 [batch_size x seq_length] 
                - tgt_msp: MSP目标 [batch_size]

        返回:
            loss_mlm: MLM损失
            loss_msp: MSP损失 
            correct_mlm: MLM正确预测的token数
            correct_msp: MSP正确预测的样本数
            denominator: 被掩码的token总数
        """

        # Masked language model (MLM).
        assert type(tgt) == tuple
        tgt_mlm, tgt_msp = tgt[0], tgt[1]       #tgt是一个元组传进来的，tgt_mlm不用修改，只需要修改tgt_msp
        loss_mlm, correct_mlm, denominator = self.mlm(memory_bank, tgt_mlm) #MLM损失，这个可以不修改，直接使用

        # Mixed sentence prediction (MSP).
        output_msp = torch.tanh(self.msp_linear_1(memory_bank[:, 0, :]))
        output_msp = self.msp_linear_2(output_msp)
        loss_msp = self.criterion(self.softmax(output_msp), tgt_msp)    #他映射到五个类别，我可以映射到十个类别，代表数据包的位置。
        correct_msp = self.softmax(output_msp).argmax(dim=-1).eq(tgt_msp).sum()

        return loss_mlm, loss_msp, correct_mlm, correct_msp, denominator
