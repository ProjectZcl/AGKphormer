import argparse

def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.') #初始为0
    parser.add_argument('--k_folds', type=int, default=5, help='k_folds.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')  # weight_decay=5e-4# 权重衰减
    parser.add_argument('--hidden_channels', type=int, default=64,  help='Number of hidden units.')
    parser.add_argument("--out_channels", type=int, default=128, help="out-channels of cnn. Default is 256.")
    parser.add_argument("--num_intervals", type=int, default=5, help=".")
    parser.add_argument("--heads", type=int, default=1, help="attention-heads")
    parser.add_argument("--dropout_trans", type=float, default=0, help="attention-dropout")
    parser.add_argument("--k", type=int, default=3, help="k")
    parser.add_argument("--num_gcn_layers", type=int, default=2, help="gcn_layers.")
    parser.add_argument("--num_transformer_layers", type=int, default=1, help="transformer_layers.")
    parser.add_argument("--num_kan_layers", type=int, default=1, help="layers2.")#默认设置了一层
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--nheads', type=float, default=1, help='Number of head attentions.')

    return parser.parse_args()