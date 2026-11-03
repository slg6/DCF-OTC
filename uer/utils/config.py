import json
from argparse import Namespace


def load_hyperparam(args):
    with open(args.config_path, mode="r", encoding="utf-8") as f:
        param = json.load(f)
    #param:{'emb_size': 768, 'feedforward_size': 3072, 'hidden_size': 768, 
    # 'hidden_act': 'gelu', 'heads_num': 12, 'layers_num': 12, 'max_seq_length': 512, 'dropout': 0.1}
    args_dict = vars(args)
    args_dict.update(param)
    args = Namespace(**args_dict)

    return args
