"""Tiny training script — the same file runs locally and in the cloud."""
import platform
from datetime import datetime, timezone
import numpy as np
import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print(f"Running on : {platform.node()} ({platform.system()})")
print(f"Started at : {datetime.now(timezone.utc).isoformat()}")

X, y = load_iris(return_X_y=True)
# Real datasets are noisy; a fixed seed keeps plain iris from saturating near
# 100% accuracy on every split, and stays fully reproducible across machines.
rng = np.random.RandomState(7)
flip = rng.choice(len(y), size=25, replace=False)
y[flip] = (y[flip] + 1) % 3
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print(f"Accuracy   : {acc:.3f}")

joblib.dump(model, "model.pkl")
print("Saved model to ./model.pkl")