# autoencoder_anomaly.py
import numpy as np
from tensorflow.keras import layers, Model, Input

def build_seq_autoencoder(timesteps=8):
    inp = Input(shape=(timesteps, 1))
    x = layers.Conv1D(16, 3, activation='relu', padding='same')(inp)
    x = layers.MaxPool1D(2)(x)
    x = layers.Conv1D(8, 3, activation='relu', padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation='relu')(x)
    # decoder
    x = layers.Dense((timesteps//2)*8, activation='relu')(x)
    x = layers.Reshape((timesteps//2, 8))(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(1, 3, activation='sigmoid', padding='same')(x)
    return Model(inp, x, name='seq_autoencoder')

if __name__ == '__main__':
    ndvi = np.load('data/ndvi.npy')  # (n, timesteps)
    # reshape
    X = ndvi[...,None].astype('float32')
    # train/test split
    n = X.shape[0]
    train_n = int(0.8*n)
    X_train = X[:train_n]
    X_val = X[train_n:]
    ae = build_seq_autoencoder(timesteps=X.shape[1])
    ae.compile(optimizer='adam', loss='mse')
    ae.fit(X_train, X_train, validation_data=(X_val,X_val), epochs=30, batch_size=16, verbose=2)
    ae.save('ndvi_autoencoder.h5')
    # compute reconstruction errors to set threshold
    recon = ae.predict(X_train)
    mse = ((recon - X_train)**2).mean(axis=(1,2))
    thresh = np.mean(mse) + 3*np.std(mse)
    print("Anomaly threshold:", thresh)
