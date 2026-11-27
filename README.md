# ✦ Particle Manipulator — Hand-Tracked CV Edition ✦

An interactive 3D particle morphing engine (150,000 WebGL particles) now controlled by **real-time hand gestures** via MediaPipe, with a trained MLP classifier, smoothing, HUD, and evaluation tools.

Original project by [Naini Vivekanand](https://github.com/kvik0802) — MIT License. Extended for a Computer Vision course project.

## Features
- 150K particles, single draw call, custom GLSL shaders, morphing between 8 shapes + text
- MediaPipe HandLandmarker (2 hands) + GestureRecognizer, GPU delegate
- Finger counting, fist, pinch, swipe, circle motion, two-hand interaction
- One-Euro smoothing, debouncing, hysteresis, 500 ms cooldown
- Trained MLP classifier (scikit-learn) exported to browser, runs in pure JS
- HUD: gesture label, confidence, FPS, latency, cheat sheet, gesture→text mode
- Evaluation panel with accuracy table + confusion matrix
- Auto-LOD for low FPS, 640×480 camera, detection ~30 FPS separate from render

## Getting Started
```bash
# Python 3.9-3.12
cd backend
pip install -r requirements.txt
# optional: train classifier
python train_classifier.py
cd ..
python backend/server.py
```
Open http://localhost:5001. Click **Enable Camera & Hand Tracking** (requires a click for camera permission).

## Optional: train your own classifier
Record samples with the HUD "Record Sample" button, export JSON, then:
```bash
python backend/train_classifier.py --samples samples.json
```
This regenerates `frontend/model.json`.

## Controls
- Mouse: drag rotate, scroll zoom, click burst
- Toggle Mouse/Gesture in HUD
- Gestures: see on-screen cheat sheet

## Evaluation
Use the Evaluation panel to run 20-50 trials per gesture; export CSV for accuracy table & confusion matrix.

## Credits
- Original: Naini Vivekanand (MIT)
- MediaPipe, Three.js, scikit-learn

## License
MIT