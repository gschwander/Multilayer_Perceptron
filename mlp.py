import argparse
from SRC.split import run_split
from SRC.train import run_train
from SRC.predict import run_predict

MODES = {
    'split': run_split,
    'train': run_train,
    'predict': run_predict
}


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode', required=True)

    split_parser = subparsers.add_parser('split')
    split_parser.add_argument('--dataset', type=str, default='data/data.csv')
    split_parser.add_argument('--seed', type=int, default=42)

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--layer', type=int, nargs='+', default=[24, 24])
    train_parser.add_argument('--epochs', type=int, default=100)
    train_parser.add_argument('--loss', type=str, default='categoricalCrossentropy', choices=['categoricalCrossentropy'])
    train_parser.add_argument('--batch_size', type=int, default=8)
    train_parser.add_argument("--learning_rate", type=float, default=0.01)
    train_parser.add_argument('--early_stopping', action='store_true')
    train_parser.add_argument('--patience', type=int, default=10)

    predict_parser = subparsers.add_parser('predict')
    predict_parser.add_argument("--dataset", type=str, default='data/data_valid.csv')
    predict_parser.add_argument("--model", type=str, default="models/saved_model.npy")

    args = parser.parse_args()
    MODES[args.mode](args)

if __name__ == '__main__':
    main()
