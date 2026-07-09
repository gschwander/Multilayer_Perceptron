import os
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

def save_model(weight, biases, layer_size, mean, std, path="models/saved_model.npy"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model = {
        "weight": weight,
        "biases": biases,
        "layer_size": layer_size,
        "mean": mean,
        "std": std,
    }
    np.save(path, model, allow_pickle=True)
    print(f"> saving model '{path}' to disk...")


def binary_cross_entropy(y_true, p_pred):
    p_pred_clipped = np.clip(p_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(p_pred_clipped) + (1 - y_true) * np.log(1 - p_pred_clipped))


def save_metrics(train_loss, train_acc, valid_loss, valid_acc, path='models/metrics.csv'):
    df = pd.DataFrame({
        'train_loss': train_loss,
        'valid_loss': valid_loss,
        'train_accurency': train_acc,
        'valid_accurency': valid_acc
    })
    df.to_csv(path, index=False)
