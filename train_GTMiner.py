import argparse
from train import train_model
import config

parser = argparse.ArgumentParser(description='GTMiner')
parser.add_argument("--city", type=str, default='sin', help='City dataset (sin, tor, sea, mel)')
parser.add_argument("--lm", type=str, default='bert')
parser.add_argument("--lr", type=float, default=3e-5)
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--max_len", type=int, default=128)
parser.add_argument("--device", type=str, default='cpu')
parser.add_argument("--finetuning", dest='finetuning', action='store_true')
parser.add_argument("--do_extend", dest='do_extend', action='store_true')
parser.add_argument("--do_repair", dest='do_repair', action='store_true')
parser.add_argument("--save_model", dest='save_model', action='store_true')

hp = parser.parse_args()

train_model(hp)
