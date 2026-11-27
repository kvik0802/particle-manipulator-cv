#!/usr/bin/env python3
"""Train a small MLP gesture classifier from recorded landmark samples.

Usage:
  python backend/train_classifier.py                      # uses synthetic bootstrap data
  python backend/train_classifier.py --samples samples.json  # uses recorded data
Outputs: frontend/model.json
"""
import argparse, json, os, numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Canonical open-palm landmarks (21 points) in a simple local frame.
BASE_HAND = np.array([
    [0.0,0.0,0.0],   # 0 wrist
    [-0.2,0.1,0.0],  # 1 thumb cmc
    [-0.35,0.25,0.0],# 2 thumb mcp
    [-0.45,0.4,0.0], # 3 thumb ip
    [-0.5,0.55,0.0], # 4 thumb tip
    [-0.1,0.45,0.0], # 5 index mcp
    [-0.1,0.7,0.0],  # 6 index pip
    [-0.1,0.85,0.0], # 7 index dip
    [-0.1,1.0,0.0],  # 8 index tip
    [0.05,0.45,0.0], # 9 middle mcp
    [0.05,0.72,0.0], # 10 middle pip
    [0.05,0.88,0.0], # 11 middle dip
    [0.05,1.05,0.0], # 12 middle tip
    [0.2,0.42,0.0],  # 13 ring mcp
    [0.2,0.68,0.0],  # 14 ring pip
    [0.2,0.83,0.0],  # 15 ring dip
    [0.2,0.98,0.0],  # 16 ring tip
    [0.35,0.38,0.0], # 17 pinky mcp
    [0.35,0.6,0.0],  # 18 pinky pip
    [0.35,0.73,0.0], # 19 pinky dip
    [0.35,0.85,0.0], # 20 pinky tip
], dtype=float)

def curl(hand, fingers):
    h = hand.copy()
    # map finger->indices
    fmap = {'thumb':[1,2,3,4], 'index':[5,6,7,8], 'middle':[9,10,11,12],
            'ring':[13,14,15,16], 'pinky':[17,18,19,20]}
    for name in fingers:
        for idx in fmap[name][1:]:
            h[idx,1] -= 0.35  # curl down
            h[idx,0] *= 0.7
    return h

def make_gesture(name, n=200):
    samples = []
    for _ in range(n):
        h = BASE_HAND.copy()
        if name == 'open' or name == 'five':
            pass
        elif name == 'fist':
            h = curl(h, ['thumb','index','middle','ring','pinky'])
        elif name == 'one':
            h = curl(h, ['thumb','middle','ring','pinky'])
        elif name == 'two':
            h = curl(h, ['thumb','ring','pinky'])
        elif name == 'three':
            h = curl(h, ['thumb','pinky'])
        elif name == 'four':
            h = curl(h, ['thumb'])
        elif name == 'thumbs_up':
            h = curl(h, ['index','middle','ring','pinky'])
            # thumb up
            h[4,1] += 0.2
        elif name == 'peace':
            h = curl(h, ['thumb','ring','pinky'])
            # V shape: spread index & middle
            h[8,0] -= 0.15; h[12,0] += 0.15
        elif name == 'point':
            h = curl(h, ['thumb','middle','ring','pinky'])
        # add noise
        h = h + np.random.normal(0, 0.03, h.shape)
        # normalize relative to wrist, scaled by hand size (match browser)
        wrist = h[0].copy()
        mid = h[9].copy()
        scale = np.linalg.norm(mid - wrist) or 1.0
        norm = ((h - wrist) / scale).flatten().tolist()
        samples.append(norm)
    return samples

def load_samples(path):
    with open(path) as f:
        data = json.load(f)
    X, y = [], []
    for s in data:
        X.append(s['features'])
        y.append(s['label'])
    return np.array(X), np.array(y)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', default=None)
    args = ap.parse_args()

    if args.samples and os.path.exists(args.samples):
        X, y = load_samples(args.samples)
        print('Loaded', len(X), 'recorded samples')
    else:
        labels = ['fist','open','one','two','three','four','five','thumbs_up','peace','point']
        # map open->open, five alias
        X_list, y_list = [], []
        for lab in labels:
            feats = make_gesture(lab, n=250)
            X_list.extend(feats)
            y_list.extend([lab]*len(feats))
        X = np.array(X_list); y = np.array(y_list)
        print('Generated synthetic dataset:', X.shape)

    # normalize labels: treat 'five' as 'open'
    y = np.array(['open' if (str(v)=='five') else str(v) for v in y])

    le = LabelEncoder()
    yenc = le.fit_transform(y)
    labels = list(le.classes_)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    Xtr, Xte, ytr, yte = train_test_split(Xs, yenc, test_size=0.2, random_state=42, stratify=yenc)
    clf = MLPClassifier(hidden_layer_sizes=(64,32), activation='relu', max_iter=500, random_state=42)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = accuracy_score(yte, pred)
    print('Test accuracy:', round(acc*100,2),'%')
    print('Confusion matrix:\n', confusion_matrix(yte, pred))

    # export weights
    weights = [w.tolist() for w in clf.coefs_]
    biases = [b.tolist() for b in clf.intercepts_]
    model = {
        'labels': labels,
        'mean': scaler.mean_.tolist(),
        'std': np.sqrt(scaler.var_).tolist(),
        'weights': weights,
        'biases': biases,
    }
    out_path = os.path.join(BASE, 'frontend', 'model.json')
    with open(out_path, 'w') as f:
        json.dump(model, f)
    print('Saved model to', out_path)

if __name__ == '__main__':
    main()
