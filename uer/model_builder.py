import torch
from uer.layers import *
from uer.encoders import *
from uer.targets import *
from uer.models.model import Model


def build_model(args):
    """
    Build universial encoder representations models.
    The combinations of different embedding, encoder, 
    and target layers yield pretrained models of different 
    properties. 
    We could select suitable one for downstream tasks.
    """

    embedding = str2embedding[args.embedding](args, len(args.vocab))    #WordPosSegEmbedding(args, len(args.vocab)) # 词嵌入层
    encoder = str2encoder[args.encoder](args)   #TransformerEncoder(args) # 编码器
    target = str2target[args.target](args, len(args.vocab)) #BertFlowTarget(args, len(args.vocab)) # 目标层
    model = Model(args, embedding, encoder, target)     #model是由Model类构造的，forward方法也在这个类中被定义。
    """
    Pretraining models consist of three parts:
        - embedding
        - encoder
        - target
    """
    return model
