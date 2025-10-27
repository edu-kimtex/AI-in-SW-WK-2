# inference_demo.py
import numpy as np
import joblib
from tensorflow.keras.models import load_model

ADVISORY_MAP = {
  0: "No action needed. Monitor weekly.",
  1: "Low NDVI detected: check irrigation and soil moisture immediately.",
  2: "Possible pest damage: send photo; consider local extension for ID.",
  3: "Nutrient deficiency suspected: soil test recommended; consider applying fertilizer."
}

# load artifacts
yield_model = load_model('yield_model_final.h5', compile=False)
ae = load_model('ndvi_autoencoder.h5', compile=False)
clf = joblib.load('advisory_clf.joblib')

# load a random field
images = np.load('data/images.npy')
ndvi = np.load('data/ndvi.npy')
weather = np.load('data/weather.npy')
meta = __import__('pandas').read_csv('data/meta.csv')
i = 0  # index of field to test

img_seq = images[i:i+1]  # shape (1, t, H, W, C)
weather_seq = weather[i:i+1]
meta_vec = meta[['soil_quality','planting_offset']].values[i:i+1].astype('float32')

pred_yield = yield_model.predict([img_seq, weather_seq, meta_vec])[0,0]
print(f"Predicted yield (units): {pred_yield:.3f}")

# anomaly detection
x = ndvi[i:i+1, :, None].astype('float32')
recon = ae.predict(x)
mse = ((recon - x)**2).mean()
print(f"NDVI reconstruction MSE: {mse:.6f}")

# To compute threshold, we approximate from training data (simplified)
train_mse = ((ae.predict(ndvi[:int(0.8*len(ndvi)), :, None]) - ndvi[:int(0.8*len(ndvi)), :, None])**2).mean(axis=(1,2))
thresh = train_mse.mean() + 3*train_mse.std()
print("Threshold (approx):", thresh)
if mse > thresh:
    print("Anomaly flagged for this field.")
else:
    print("No anomaly flagged.")

# advisory classifier
feat = np.array([[ndvi[i].mean(), ndvi[i].std(), meta['soil_quality'].iloc[i]]])
label = clf.predict(feat)[0]
print("Advisory:", ADVISORY_MAP[label])
