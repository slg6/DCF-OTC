import torch
import torch.nn as nn
from uer.targets import *


# class BertPacketsTarget(MlmTarget):
#     """
#     BERT流量分类目标层:
#     1. MLM(掩码语言模型): 预测被掩码的token
#     2. MSP(混合句子预测): 预测流量序列的类别
#     """

#     def __init__(self, args, vocab_size):
#         super(BertPacketsTarget, self).__init__(args, vocab_size)
#         # MSP.
#         self.msp_linear_1 = nn.Linear(args.hidden_size, args.hidden_size)
#         self.msp_linear_2 = nn.Linear(args.hidden_size, 4)
#         # self.pos_linear = nn.Linear(args.hidden_size, 10)
#     def forward(self, memory_bank, tgt):
#         """
#         前向传播计算损失
        
#         参数:
#             memory_bank: 编码器输出 [batch_size x seq_length x hidden_size]
#             tgt: 包含两个目标的元组:
#                 - tgt_mlm: MLM目标 [batch_size x seq_length] 
#                 - tgt_msp: MSP目标 [batch_size]

#         返回:
#             loss_mlm: MLM损失
#             loss_msp: MSP损失 
#             correct_mlm: MLM正确预测的token数
#             correct_msp: MSP正确预测的样本数
#             denominator: 被掩码的token总数
#         """

#         # Masked language model (MLM).
#         assert type(tgt) == tuple       #tgt_mlm, label, field_modified
#         tgt_mlm, label, field_modified = tgt[0], tgt[1], tgt[2]       #tgt是一个元组传进来的，tgt_mlm不用修改，field_modified与tgt_mlm相似
#         loss_mlm, correct_mlm, denominator_mlm = self.mlm(memory_bank, tgt_mlm) #MLM损失，这个可以不修改，直接使用 denominator

#         loss_fm, correct_fm, denominator_fm = self.mlm(memory_bank, field_modified)  #字段修改损失
#         # Mixed sentence prediction (MSP).
#         output_msp = torch.tanh(self.msp_linear_1(memory_bank[:, 0, :]))
#         output_msp = self.msp_linear_2(output_msp)
#         loss_msp = self.criterion(self.softmax(output_msp), label)    #他映射到五个类别，我有四种标签，分别代表不同的逆序方式。
#         correct_msp = self.softmax(output_msp).argmax(dim=-1).eq(label).sum()

#         return loss_mlm, loss_msp, loss_fm,correct_mlm, correct_msp, correct_fm, denominator_mlm, denominator_fm

class BertPacketsTarget(nn.Module):
    """
    BERT流量分类目标层:
    1. MLM(掩码语言模型): 预测被掩码的token
    2. MSP(混合句子预测): 预测流量序列的类别
    """

    def __init__(self, args, vocab_size):
        # super(BertPacketsTarget, self).__init__(args, vocab_size)
        super(BertPacketsTarget, self).__init__()   #nn.Module.__init__() 不接受任何额外参数
        # MSP.
        self.msp_linear_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.msp_linear_2 = nn.Linear(args.hidden_size, 4)
        self.mlm1 = MlmTarget(args, vocab_size)
        self.mlm2 = MlmTarget(args, vocab_size)
        self.criterion = nn.NLLLoss()
        self.softmax = nn.LogSoftmax(dim=-1)
        # self.pos_linear = nn.Linear(args.hidden_size, 10)
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
        assert type(tgt) == tuple       #tgt_mlm, label, field_modified
        tgt_mlm, label, field_modified = tgt[0], tgt[1], tgt[2]       #tgt是一个元组传进来的，tgt_mlm不用修改，field_modified与tgt_mlm相似
        loss_mlm, correct_mlm, denominator_mlm = self.mlm1(memory_bank, tgt_mlm) #MLM损失，这个可以不修改，直接使用 denominator

        loss_fm, correct_fm, denominator_fm = self.mlm2(memory_bank, field_modified)  #字段修改损失
        # Mixed sentence prediction (MSP).
        output_msp = torch.tanh(self.msp_linear_1(memory_bank[:, 0, :]))
        output_msp = self.msp_linear_2(output_msp)
        loss_msp = self.criterion(self.softmax(output_msp), label)    #他映射到五个类别，我有四种标签，分别代表不同的逆序方式。
        correct_msp = self.softmax(output_msp).argmax(dim=-1).eq(label).sum()   #计算的是 当前 batch 里预测正确的样本个数

        return loss_mlm, loss_msp, loss_fm,correct_mlm, correct_msp, correct_fm, denominator_mlm, denominator_fm
