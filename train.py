"""Train an iris classifier. Usage: python train.py --test-size 0.3"""
import argparse
import platform
from datetime import datetime, timezone

import numpy as np
import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train(test_size: float) -> float:
    X, y = load_iris(return_X_y=True)
    rng = np.random.RandomState(7)
    flip = rng.choice(len(y), size=25, replace=False)
    y[flip] = (y[flip] + 1) % 3
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    joblib.dump(model, "model.pkl")
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=float, default=0.3)
    args = parser.parse_args()

    print(f"Running on : {platform.node()} ({platform.system()})")
    print(f"Started at : {datetime.now(timezone.utc).isoformat()}")

    accuracy = train(args.test_size)
    print(f"Accuracy   : {accuracy:.3f} (test_size={args.test_size})")
    print("Saved model to ./model.pkl")