import pandas as pd
import numpy as np


def run_train(args):
    try:
        set_train = pd.read_csv("data/data_train.csv")
    except FileNotFoundError:
        print("Error: 'data/data_train.csv' not found. Run the 'split' command first.")
        return
    y_train = (set_train['diagnosis'] == 'M').astype(int)
    X_train = set_train.drop(columns=['id', 'diagnosis'])

    X_train_mean = X_train.mean(axis=0)
    X_train_std = X_train.std(axis=0)
    X_train_norm = (X_train - X_train_mean) / X_train_std

    for epoch in range(args.epochs):
        # mélanger les indices du dataset d'entraînement à chaque epoch
        indices = np.random.permutation(len(X_train))
        
        for start in range(0, len(X_train), args.batch_size):
            batch_indices = indices[start:start + args.batch_size]
            
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            
            # forward pass
            # predictions = ...
            
            # calcul de la loss
            # loss = ...
            
            # backward pass
            # gradients = ...
            
            # mise à jour des poids
            # weights = weights - learning_rate * gradients
        
        # à la fin de chaque epoch : calculer et afficher les métriques
        # train_loss, train_acc = ...
        # val_loss, val_acc = ...
        print(f"epoch {epoch+1}/{epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")