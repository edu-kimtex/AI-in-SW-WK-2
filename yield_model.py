# yield_model.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input, Model

def build_cnn_backbone(img_shape=(32,32,3)):
    inp = Input(shape=img_shape)
    x = layers.Conv2D(16, 3, activation='relu', padding='same')(inp)
    x = layers.MaxPool2D(2)(x)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    return Model(inp, x, name='cnn_backbone')

def build_temporal_model(timesteps=8, img_shape=(32,32,3), weather_dim=2, meta_dim=2):
    # per-timestep image backbone
    cnn = build_cnn_backbone(img_shape)
    # inputs
    img_seq = Input(shape=(timesteps, img_shape[0], img_shape[1], img_shape[2]), name='img_seq')
    weather_seq = Input(shape=(timesteps, weather_dim), name='weather_seq')
    meta = Input(shape=(meta_dim,), name='meta')  # e.g., soil, planting_offset

    # apply CNN across timesteps (TimeDistributed)
    td = layers.TimeDistributed(cnn)(img_seq)  # (batch, timesteps, features)
    # fuse with weather features per timestep
    fused_per_t = layers.Concatenate(axis=-1)([td, weather_seq])  # (batch, timesteps, features+weather)
    # temporal aggregator
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(fused_per_t)
    # combine with meta
    x = layers.Concatenate()([x, meta])
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    out = layers.Dense(1, activation='linear', name='yield')(x)
    model = Model(inputs=[img_seq, weather_seq, meta], outputs=out, name='yield_model')
    return model

if __name__ == '__main__':
    model = build_temporal_model()
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.summary()
