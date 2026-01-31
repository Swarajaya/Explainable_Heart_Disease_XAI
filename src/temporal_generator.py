import numpy as np

def create_temporal_sequences(X, y, seq_len=5):
    sequences = []
    labels = []

    for i in range(len(X) - seq_len):
        sequences.append(X[i:i+seq_len])
        labels.append(y.iloc[i+seq_len])

    return np.array(sequences), np.array(labels)
