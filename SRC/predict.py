import numpy as np
import pandas as pd
from SRC.utils import forward_pass, binary_cross_entropy

def run_predict(args):
    try:
        model = np.load(args.model, allow_pickle=True).item()
    except FileNotFoundError:
        print(f"Error: '{args.model}' not found. Run the 'train' command first.")
        return
    try:
        data = pd.read_csv(args.dataset)
    except FileNotFoundError:
        print(f"Error: '{args.dataset}' not found. Run the 'split' command first.")
        return
    
    print(f"Architecture: {model['layer_size']}")

    y_data = (data['diagnosis'] == 'M').astype(int)
    X_data = data.drop(columns=['id', 'diagnosis'])

    std_safe = np.where(model['std'] == 0, 1e-8, model['std'])
    X_data_norm = (X_data - model['mean']) / std_safe
    X_data_norm = X_data_norm.values
    y_data = y_data.values

    weight = model['weight']
    biases = model['biases']
    
    pred = forward_pass(X_data_norm, weight, biases)

    pred_class = np.argmax(pred, axis=1)
    accurency = np.mean(pred_class == y_data)
    bce = binary_cross_entropy(y_data, pred[:, 1])

    print(f'Accurency : {accurency}')
    print(f'Binary Cross Entropy : {bce}')

