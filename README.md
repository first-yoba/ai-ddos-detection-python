# AI DDoS Detection (Python)

A clean, beginner-friendly **machine learning pipeline in Python** that classifies network-traffic-style features as **benign (0)** or **DDoS/attack (1)**.

This repo is intentionally structured like a real project (modules, artifacts, reproducible run) rather than a single notebook.

---

## What this project does

1. Loads a CSV dataset containing numeric traffic features + a `label` column
2. Cleans the data (drops missing, keeps numeric features)
3. Scales features with `StandardScaler`
4. Trains a baseline classifier (**Logistic Regression**)
5. Prints evaluation metrics (classification report + confusion matrix)
6. Saves the trained model + scaler to disk for reuse

---

## Project structure

