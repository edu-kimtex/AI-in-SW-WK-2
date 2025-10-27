# advisory_classifier.py
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# We'll synthesize problem labels for demo:
# 0 = healthy, 1 = low_ndvi_drought, 2 = pest, 3 = nutrient_deficit
def synthesize_problem_labels(ndvi, soil):
    # naive rules to produce labels
    mean_ndvi = ndvi.mean(axis=1)
    labels = []
    for m,s in zip(mean_ndvi, soil.squeeze()):
        if m > 0.05:
            labels.append(0)
        elif m < -0.1:
            labels.append(1)  # drought
        elif s < 0.2:
            labels.append(3)  # nutrient
        else:
            labels.append(2)  # pest as fallback
    return np.array(labels)

if __name__ == '__main__':
    ndvi = np.load('data/ndvi.npy')  # (n, t)
    meta = __import__('pandas').read_csv('data/meta.csv')
    soil = meta[['soil_quality']].values.astype('float32')
    # create features: mean ndvi, ndvi std, soil
    feat = np.stack([ndvi.mean(axis=1), ndvi.std(axis=1), soil.squeeze()], axis=1)
    y = synthesize_problem_labels(ndvi, soil)
    X_train, X_test, y_train, y_test = train_test_split(feat, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print(classification_report(y_test, preds))
    joblib.dump(clf, 'advisory_clf.joblib')
    print("Saved advisory classifier to advisory_clf.joblib")
