import pandas as pd

def run_split(args):
    df = pd.read_csv(args.dataset, header=None)
    columns = ['id', 'diagnosis'] + [f'feature{i}' for i in range(30)]
    df.columns = columns

    print(df.shape)