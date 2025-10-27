# train_yield_model.py
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from yield_model import build_temporal_model

# load data
images = np.load('data/images.npy')  # (n, t, H, W, C)
weather = np.load('data/weather.npy')  # (n, t, 2)
meta_df = pd.read_csv('data/meta.csv')
meta = meta_df[['soil_quality','planting_offset']].values.astype('float32')
y = meta_df['yield'].values.astype('float32')

# simple train/test split
n = images.shape[0]
idx = np.arange(n)
np.random.shuffle(idx)
train_idx = idx[:int(0.8*n)]
val_idx = idx[int(0.8*n):]

img_train, img_val = images[train_idx], images[val_idx]
weather_train, weather_val = weather[train_idx], weather[val_idx]
meta_train, meta_val = meta[train_idx], meta[val_idx]
y_train, y_val = y[train_idx], y[val_idx]

model = build_temporal_model(timesteps=images.shape[1], img_shape=images.shape[2:], weather_dim=weather.shape[-1], meta_dim=meta.shape[1])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

callbacks = [
    EarlyStopping(patience=8, restore_best_weights=True),
    ModelCheckpoint('best_yield_model.h5', save_best_only=True, monitor='val_loss')
]

history = model.fit(
    [img_train, weather_train, meta_train],
    y_train,
    validation_data=([img_val, weather_val, meta_val], y_val),
    epochs=50,
    batch_size=16,
    callbacks=callbacks,
    verbose=2
)

# save
model.save('yield_model_final.h5')
print("Training done. Model saved as yield_model_final.h5")
