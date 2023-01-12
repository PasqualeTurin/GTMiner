import argparse
from train import train_model, search_aois
import config

parser = argparse.ArgumentParser(description='Classifier')
parser.add_argument("--city", type=str, default='sin', help='City dataset (sin, tor, sea, mel)')
parser.add_argument("--fe", type=str, default='bert')
parser.add_argument("--lr", type=float, default=3e-5)
parser.add_argument("--alpha", type=float, default=2.0)
parser.add_argument("--beta", type=float, default=1.0)
parser.add_argument("--n_epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--max_len", type=int, default=32)
parser.add_argument("--device", type=str, default='cpu')
parser.add_argument("--save_model", dest='save_model', action='store_true')

hp = parser.parse_args()

search_aois(hp)
