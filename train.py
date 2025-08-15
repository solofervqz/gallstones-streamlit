# train.py
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import numpy as np
import joblib
from features import convert_commas_to_dots, make_features, split_X_y

DATA_PATH = "data/dataset-uci.csv"
MODEL_PATH = "model.joblib"      # guardaremos {'pipeline': pipeline, 'threshold': t}

def tune_global_threshold(model, X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    thresholds = np.arange(0.1, 0.91, 0.01)

    best_t, best_f1 = 0.5, -1
    for t in thresholds:
        f1s = []
        for tr, te in skf.split(X, y):
            model.fit(X[tr], y[tr])
            proba = model.predict_proba(X[te])[:, 1]
            pred = (proba >= t).astype(int)
            f1s.append(f1_score(y[te], pred, zero_division=0))
        m = np.mean(f1s)
        if m > best_f1:
            best_f1, best_t = m, t
    return best_t, best_f1

def main():
    # 1) Carga y conversiones
    df = pd.read_csv(DATA_PATH)
    df = convert_commas_to_dots(df)

    # 2) Feature engineering
    df_feat = make_features(df)
    X, y = split_X_y(df_feat)

    # 3) Pipeline (imputer → scaler → LR con tus mejores params)
    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(penalty="l1", solver="liblinear", C=1.0, max_iter=1000, random_state=42))
    ])

    # 4) Para buscar threshold global óptimo usamos la salida antes del ajuste final
    # Convertimos X,y a numpy por si vienen como DataFrame/Series
    X_np = X.to_numpy()
    y_np = y.to_numpy().astype(int)

    best_t, best_f1 = tune_global_threshold(pipe, X_np, y_np)
    print(f"Mejor threshold: {best_t:.2f} | F1 CV: {best_f1:.4f}")

    # 5) Ajuste final con todos los datos
    pipe.fit(X_np, y_np)

    # 6) Guardar pipeline + threshold
    joblib.dump({"pipeline": pipe, "threshold": float(best_t)}, MODEL_PATH)
    print(f"Modelo guardado en {MODEL_PATH}")

if __name__ == "__main__":
    main()
