import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def forward_pass(X, weight, biases):
    A = X
    for i, (W, b) in enumerate(zip(weight, biases)):
        Z = A @ W + b
        A = softmax(Z) if i == len(weight) - 1 else sigmoid(Z)
    return A


def graph_loss_acc(train_loss, train_acc, valid_loss, valid_acc):
    plt.figure()
    plt.plot(train_loss, label="train")
    plt.plot(valid_loss, label="validation")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss")
    plt.legend()
    plt.savefig("graph/loss.png")

    plt.figure()
    plt.plot(train_acc, label="train")
    plt.plot(valid_acc, label="validation")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Accuracy")
    plt.legend()
    plt.savefig("graph/accuracy.png")

def run_train(args):
    try:
        set_train = pd.read_csv("data/data_train.csv")
    except FileNotFoundError:
        print("Error: 'data/data_train.csv' not found. Run the 'split' command first.")
        return
    try:
        set_valid = pd.read_csv("data/data_valid.csv")
    except FileNotFoundError:
        print("Error: 'data/data_valid.csv' not found. Run the 'split' command first.")
        return
    y_train = (set_train['diagnosis'] == 'M').astype(int)
    y_valid = (set_valid['diagnosis'] == 'M').astype(int)
    X_train = set_train.drop(columns=['id', 'diagnosis'])
    X_valid = set_valid.drop(columns=['id', 'diagnosis'])

    X_train_mean = X_train.mean(axis=0)
    X_train_std = X_train.std(axis=0)
    X_train_norm = (X_train - X_train_mean) / X_train_std
    X_valid_norm = (X_valid - X_train_mean) / X_train_std
    
    
    X_train_norm = X_train_norm.values
    X_valid_norm = X_valid_norm.values
    y_train = y_train.values
    y_valid = y_valid.values

    layer_size = [X_train.shape[1]] + args.layer + [2]

    weight = []
    biases = []
    for n_in, n_out in zip(layer_size[:-1], layer_size[1:]):
        limit = np.sqrt(6 / n_in)
        W = np.random.uniform(-limit, limit, (n_in, n_out))
        weight.append(W)
        b = np.zeros(n_out)
        biases.append(b)

    train_loss_lst = []
    train_acc_lst = []
    valid_loss_lst = []
    valid_acc_lst = []

    for epoch in range(args.epochs):
        # mélanger les indices du dataset d'entraînement à chaque epoch
        indices = np.random.permutation(len(X_train_norm))
        
        for start in range(0, len(X_train), args.batch_size):
            batch_indices = indices[start:start + args.batch_size]
            
            X_batch = X_train_norm[batch_indices]
            y_batch = y_train[batch_indices]
            
            # forward pass
            A = X_batch
            Acti = [A]

            for i, (W, b) in enumerate(zip(weight, biases)):
                Z = A @ W + b

                if i == len(weight) - 1:
                    A = softmax(Z)
                else:
                    A = sigmoid(Z)
                Acti.append(A)

            
            # calcul de la loss
            y_onehot = np.eye(2)[y_batch]
            A_clipped = np.clip(A, 1e-15, 1 - 1e-15)
            loss = -np.mean(np.sum(y_onehot * np.log(A_clipped), axis=1))
            
            # backward pass
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
                dW_tmp = args.learning_rate * dW
                db_tmp = args.learning_rate * db

                w -= dW_tmp
                b -= db_tmp

        # à la fin de chaque epoch : calculer et afficher les métriques

        train_pred = forward_pass(X_train_norm, weight, biases)
        val_pred = forward_pass(X_valid_norm, weight, biases)
        
        y_onehot_train = np.eye(2)[y_train]
        X_train_clipped = np.clip(train_pred, 1e-15, 1 - 1e-15)
        train_loss = -np.mean(np.sum(y_onehot_train * np.log(X_train_clipped), axis=1))
        
        y_onehot_valid = np.eye(2)[y_valid]
        X_valid_clipped = np.clip(val_pred, 1e-15, 1 - 1e-15)
        valid_loss = -np.mean(np.sum(y_onehot_valid * np.log(X_valid_clipped), axis=1))
        
        train_pred_class = np.argmax(train_pred, axis=1)
        train_acc = np.mean(train_pred_class == y_train)

        valid_pred_class = np.argmax(val_pred, axis=1)
        valid_acc = np.mean(valid_pred_class == y_valid)

        print(f"epoch {epoch+1}/{args.epochs} - loss: {train_loss:.4f} - acc: {train_acc:.4f} - val_loss: {valid_loss:.4f} - val_acc: {valid_acc:.4f}")

        train_loss_lst.append(train_loss)
        train_acc_lst.append(train_acc)
        valid_loss_lst.append(valid_loss)
        valid_acc_lst.append(valid_acc)
    
    graph_loss_acc(train_loss_lst, train_acc_lst, valid_loss_lst, valid_acc_lst)