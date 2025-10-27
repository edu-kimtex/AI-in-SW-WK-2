# generate_synthetic_data.py
import numpy as np
import pandas as pd
import os

def make_synthetic_dataset(n_fields=200, timesteps=8, img_size=32, seed=42):
    np.random.seed(seed)
    images = np.random.rand(n_fields, timesteps, img_size, img_size, 3).astype('float32')  # satellite RGB
    # simulate NDVI-ish series by averaging green-red band difference
    ndvi = (images[...,1] - images[...,0]).mean(axis=(2,3))  # shape (n_fields, timesteps)
    # synthetic weather (temp, rainfall)
    weather = np.random.normal(loc=0, scale=1, size=(n_fields, timesteps, 2)).astype('float32')
    # metadata: soil_quality (0..1), planting_date offset
    soil = np.random.rand(n_fields, 1).astype('float32')
    planting_offset = np.random.randint(0, 10, size=(n_fields,1)).astype('float32')
    # yield label: function of mean NDVI, soil, and noise
    mean_ndvi = ndvi.mean(axis=1)
    yield_t = (2.5 * mean_ndvi + 1.8 * soil.squeeze() + np.random.normal(0,0.2,size=n_fields)).astype('float32')
    # create a simple dataframe for metadata per field
    df = pd.DataFrame({
        'field_id': np.arange(n_fields),
        'soil_quality': soil.squeeze(),
        'planting_offset': planting_offset.squeeze(),
        'yield': yield_t
    })
    os.makedirs('data', exist_ok=True)
    np.save('data/images.npy', images)
    np.save('data/ndvi.npy', ndvi)
    np.save('data/weather.npy', weather)
    df.to_csv('data/meta.csv', index=False)
    print('Saved dataset to data/ (images.npy, ndvi.npy, weather.npy, meta.csv)')
    return images, ndvi, weather, df

if __name__ == '__main__':
    make_synthetic_dataset()
