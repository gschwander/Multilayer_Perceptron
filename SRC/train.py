import pandas as pd
import numpy as np
from SRC.utils import sigmoid, softmax, forward_pass, graph_loss_acc, save_model, save_metrics


def load_data(args):
    set_train = pd.read_csv("data/data_train.csv")
    set_valid = pd.read_csv("data/data_valid.csv")

    y_train = (set_train['diagnosis'] == 'M').astype(int).values
    y_valid = (set_valid['diagnosis'] == 'M').astype(int).values
    X_train = set_train.drop(columns=['id', 'diagnosis'])
    X_valid = set_valid.drop(columns=['id', 'diagnosis'])

    X_train_mean = X_train.mean(axis=0)
    X_train_std = X_train.std(axis=0)
    X_train_norm = ((X_train - X_train_mean) / X_train_std).values
    X_valid_norm = ((X_valid - X_train_mean) / X_train_std).values

    return X_train_norm, y_train, X_valid_norm, y_valid, X_train_mean.values, X_train_std.values


def init_network(n_features, hidden_layers, n_classes=2):
    layer_size = [n_features] + hidden_layers + [n_classes]

    weight = []
    biases = []
    for n_in, n_out in zip(layer_size[:-1], layer_size[1:]):
        limit = np.sqrt(6 / n_in)
        weight.append(np.random.uniform(-limit, limit, (n_in, n_out)))
        biases.append(np.zeros(n_out))

    return weight, biases, layer_size


def forward_backward(X_batch, y_batch, weight, biases, learning_rate):
    # forward pass
    A = X_batch
    Acti = [A]

    for i, (W, b) in enumerate(zip(weight, biases)):
        Z = A @ W + b
        A = softmax(Z) if i == len(weight) - 1 else sigmoid(Z)
        Acti.append(A)

    # backward pass
    y_onehot = np.eye(2)[y_batch]
    dW_list = []
    db_list = []

    dZ = A - y_onehot

    for i in reversed(range(len(weight))):
        A_prev = Acti[i]
        dW = A_prev.T @ dZ
        db = np.sum(dZ, axis=0)

        dW_list.insert(0, dW)
        db_list.insert(0, db)

        if i > 0:
            dA_prev = dZ @ weight[i].T
            dZ = dA_prev * A_prev * (1 - A_prev)

    # mise à jour des poids
    for w, b, dW, db in zip(weight, biases, dW_list, db_list):
        w -= learning_rate * dW
        b -= learning_rate * db


def evaluate(X, y, weight, biases):
    pred = forward_pass(X, weight, biases)

    y_onehot = np.eye(2)[y]
    pred_clipped = np.clip(pred, 1e-15, 1 - 1e-15)
    loss = -np.mean(np.sum(y_onehot * np.log(pred_clipped), axis=1))

    pred_class = np.argmax(pred, axis=1)
    acc = np.mean(pred_class == y)

    return loss, acc


def run_train(args):
    try:
        X_train, y_train, X_valid, y_valid, mean, std = load_data(args)
    except FileNotFoundError as e:
        print(f"Error: {e.filename} not found. Run the 'split' command first.")
        return

    weight, biases, layer_size = init_network(X_train.shape[1], args.layer)

    train_loss_lst, train_acc_lst = [], []
    valid_loss_lst, valid_acc_lst = [], []

    best_valid_loss = float('inf')
    patience_counter = 0
    best_weight = [w.copy() for w in weight]
    best_biases = [b.copy() for b in biases]

    for epoch in range(args.epochs):
        indices = np.random.permutation(len(X_train))

        for start in range(0, len(X_train), args.batch_size):
            batch_indices = indices[start:start + args.batch_size]
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]

            forward_backward(X_batch, y_batch, weight, biases, args.learning_rate)

        train_loss, train_acc = evaluate(X_train, y_train, weight, biases)
        valid_loss, valid_acc = evaluate(X_valid, y_valid, weight, biases)

        print(f"epoch {epoch+1}/{args.epochs} - loss: {train_loss:.4f} - acc: {train_acc:.4f} "
              f"- val_loss: {valid_loss:.4f} - val_acc: {valid_acc:.4f}")

        train_loss_lst.append(train_loss)
        train_acc_lst.append(train_acc)
        valid_loss_lst.append(valid_loss)
        valid_acc_lst.append(valid_acc)

        if args.early_stopping:
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                patience_counter = 0
                best_weight = [w.copy() for w in weight]
                best_biases = [b.copy() for b in biases]
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping à l'epoch {epoch+1} (patience de {args.patience} atteinte)")
                    break

    if args.early_stopping:
        weight = best_weight
        biases = best_biases

    graph_loss_acc(train_loss_lst, train_acc_lst, valid_loss_lst, valid_acc_lst)
    save_model(weight, biases, layer_size, mean, std)
    save_metrics(train_loss_lst, train_acc_lst, valid_loss_lst, valid_acc_lst)