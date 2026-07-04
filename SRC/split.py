import pandas as pd
import numpy as np

def run_split(args):
    df = pd.read_csv(args.dataset, header=None)
    columns = ['id', 'diagnosis'] + [f'feature{i}' for i in range(df.shape[1] - 2)]
    df.columns = columns

    df_m = df[df['diagnosis'] == 'M']
    df_b = df[df['diagnosis'] == 'B']

    np.random.seed(args.seed)
    indices_m = np.random.permutation(df_m.shape[0])
    indices_b = np.random.permutation(df_b.shape[0])
    split_index_m = int(0.8 * df_m.shape[0])
    split_index_b = int(0.8 * df_b.shape[0])
    train_df_m = df_m.iloc[indices_m[:split_index_m]]
    train_df_b = df_b.iloc[indices_b[:split_index_b]]
    test_df_m = df_m.iloc[indices_m[split_index_m:]]
    test_df_b = df_b.iloc[indices_b[split_index_b:]]

    train_df = pd.concat([train_df_b, train_df_m])
    test_df = pd.concat([test_df_b, test_df_m])

    train_df = train_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # print(train_df['diagnosis'].value_counts(normalize=True))
    # print(test_df['diagnosis'].value_counts(normalize=True))

    train_df.to_csv("data/data_train.csv", index=False)
    test_df.to_csv("data/data_valid.csv", index=False)