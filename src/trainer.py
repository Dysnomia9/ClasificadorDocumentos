"""
Módulo de entrenamiento de modelos.

CORRECCIÓN v2 — Partición basada en columna 'split' del CSV
============================================================
Si dataset.csv contiene la columna 'split' (generada por realistic_corpus.py),
se usa esa asignación directamente en lugar de train_test_split().
Esto garantiza que los documentos de test fueron generados con frases
distintas a las de entrenamiento, eliminando la fuga de datos.

Si la columna no existe (compatibilidad con datasets anteriores),
se cae al comportamiento original con train_test_split estratificado.

Implementa:
  - TF-IDF Vectorizer (unigramas + bigramas)
  - Multinomial Naive Bayes
  - Linear SVM (LinearSVC)
  - Comparación con GridSearchCV para hiperparámetros
  - Serialización de modelos con joblib
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, GroupKFold, cross_val_score
)
from sklearn.metrics import (
    classification_report, accuracy_score
)

from config import (
    DATASET_CSV, NB_MODEL_PKL, SVM_MODEL_PKL,
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, TEST_SIZE, RANDOM_STATE,
    MODELS_DIR
)


# ─── Pipelines ────────────────────────────────────────────────────────────────

def construir_pipeline_nb() -> Pipeline:
    """Pipeline: TF-IDF → Multinomial Naive Bayes."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features  = TFIDF_MAX_FEATURES,
            ngram_range   = TFIDF_NGRAM_RANGE,
            sublinear_tf  = True,
            min_df        = 2,
            strip_accents = "unicode",
        )),
        ("clf", MultinomialNB()),
    ])


def construir_pipeline_svm() -> Pipeline:
    """Pipeline: TF-IDF → Linear SVM."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features  = TFIDF_MAX_FEATURES,
            ngram_range   = TFIDF_NGRAM_RANGE,
            sublinear_tf  = True,
            min_df        = 2,
            strip_accents = "unicode",
        )),
        ("clf", LinearSVC(
            max_iter     = 2000,
            random_state = RANDOM_STATE,
        )),
    ])


# ─── Grillas de hiperparámetros ───────────────────────────────────────────────

GRID_NB = {
    "tfidf__max_features": [3000, 5000],
    "tfidf__ngram_range":  [(1, 1), (1, 2)],
    "clf__alpha":          [0.01, 0.1, 0.5, 1.0],
}

GRID_SVM = {
    "tfidf__max_features": [3000, 5000],
    "tfidf__ngram_range":  [(1, 1), (1, 2)],
    "clf__C":              [0.1, 1.0, 5.0, 10.0],
}


# ─── Partición train/test ─────────────────────────────────────────────────────

def _partir_dataset(df: pd.DataFrame, verbose: bool) -> tuple:
    """
    Divide el DataFrame en train y test.

    Estrategia preferida: usar columna 'split' generada por realistic_corpus.py.
    Soporta dos conjuntos de prueba:
      - test      (holdout IID)
      - test_ood  (holdout más desafiante)

    Fallback: train_test_split estratificado si no existe la columna.
    """
    if "split" in df.columns and df["split"].notna().all():
        mask_train = df["split"] == "train"
        mask_test  = df["split"] == "test"
        mask_ood   = df["split"] == "test_ood"

        X_train = df.loc[mask_train, "texto_limpio"].to_numpy(dtype=str)
        y_train = df.loc[mask_train, "clase"].to_numpy(dtype=str)
        X_test  = df.loc[mask_test,  "texto_limpio"].to_numpy(dtype=str)
        y_test  = df.loc[mask_test,  "clase"].to_numpy(dtype=str)
        X_ood   = df.loc[mask_ood,   "texto_limpio"].to_numpy(dtype=str)
        y_ood   = df.loc[mask_ood,   "clase"].to_numpy(dtype=str)

        groups_train = None
        if "source_id" in df.columns:
            groups_train = df.loc[mask_train, "source_id"].to_numpy(dtype=str)

        if verbose:
            print(f"\n  ✓ Partición desde columna 'split' del CSV (sin fuga de datos)")
            print(f"    Train: {len(X_train)} docs | Test-IID: {len(X_test)} docs | Test-OOD: {len(X_ood)} docs")
            for c in sorted(set(y_train)):
                n_tr = (y_train == c).sum()
                n_te = (y_test  == c).sum()
                n_od = (y_ood   == c).sum()
                print(f"      {c:12s}: {n_tr} train  |  {n_te} test  |  {n_od} ood")
    else:
        if verbose:
            print(f"\n  ⚠  Columna 'split' no encontrada — usando train_test_split "
                  f"({int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)})")
            print("     Para resultados honestos, regenera con: python main.py realistic")

        X_all = df["texto_limpio"].to_numpy(dtype=str)
        y_all = df["clase"].to_numpy(dtype=str)
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all,
            test_size    = TEST_SIZE,
            random_state = RANDOM_STATE,
            stratify     = y_all,
        )
        X_ood = np.array([], dtype=str)
        y_ood = np.array([], dtype=str)
        groups_train = None

    return X_train, X_test, y_train, y_test, X_ood, y_ood, groups_train


# ─── Función principal de entrenamiento ──────────────────────────────────────

def entrenar(df: pd.DataFrame | None = None,
             usar_grid_search: bool = True,
             verbose: bool = True) -> dict:
    """
    Entrena Naive Bayes y SVM sobre el corpus preprocesado.

    Parámetros
    ----------
    df               : DataFrame con columnas [texto_limpio, clase, split?].
                       Si None, carga DATASET_CSV.
    usar_grid_search : si True, optimiza hiperparámetros con GridSearchCV.
    verbose          : imprime progreso.

    Retorna
    -------
    dict con claves:
        'X_train', 'X_test', 'y_train', 'y_test',
        'pipeline_nb', 'pipeline_svm',
        'y_pred_nb', 'y_pred_svm',
        'report_nb', 'report_svm',
        'cv_nb', 'cv_svm', 'acc_nb', 'acc_svm'
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Cargar datos ──────────────────────────────────────────────────────────
    if df is None:
        df = pd.read_csv(DATASET_CSV)

    if "texto_limpio" not in df.columns:
        raise ValueError("El DataFrame debe tener la columna 'texto_limpio'. "
                         "Ejecuta primero el preprocesador.")

    X_all = df["texto_limpio"].to_numpy(dtype=str)
    y_all = df["clase"].to_numpy(dtype=str)

    if verbose:
        print(f"\nDataset total: {len(X_all)} documentos, {len(set(y_all))} clases")
        for c in sorted(set(y_all)):
            print(f"  {c}: {(y_all == c).sum()} docs")

    # ── Partición train/test/OOD ──────────────────────────────────────────────
    X_train, X_test, y_train, y_test, X_ood, y_ood, groups_train = _partir_dataset(df, verbose)

    # ── Validación cruzada con grupos (si disponibles) ────────────────────────
    if groups_train is not None and len(np.unique(groups_train)) >= 5:
        cv = GroupKFold(n_splits=5)
        cv_iter = cv.split(X_train, y_train, groups=groups_train)
        cv_for_scores = list(cv_iter)
        cv_gs = cv_for_scores
        if verbose:
            print("  ✓ Validación cruzada por grupos (source_id) para mayor realismo")
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_for_scores = cv
        cv_gs = cv

    # ── Naive Bayes ───────────────────────────────────────────────────────────
    if verbose:
        print("\n── Entrenando Naive Bayes ──────────────────────────────────────")

    if usar_grid_search:
        gs_nb = GridSearchCV(construir_pipeline_nb(), GRID_NB,
                             cv=cv_gs, scoring="f1_macro", n_jobs=-1, verbose=0)
        gs_nb.fit(X_train, y_train)
        pipeline_nb = gs_nb.best_estimator_
        if verbose:
            print(f"  Mejores params NB: {gs_nb.best_params_}")
            print(f"  Mejor F1 (CV):     {gs_nb.best_score_:.4f}")
    else:
        pipeline_nb = construir_pipeline_nb()
        pipeline_nb.fit(X_train, y_train)

    cv_nb     = cross_val_score(pipeline_nb, X_train, y_train,
                                cv=cv_for_scores, scoring="f1_macro", n_jobs=-1)
    y_pred_nb = pipeline_nb.predict(X_test)
    report_nb = classification_report(y_test, y_pred_nb,
                                      target_names=sorted(set(y_all)), digits=4)
    acc_nb    = accuracy_score(y_test, y_pred_nb)

    y_pred_nb_ood = np.array([], dtype=str)
    acc_nb_ood = np.nan
    report_nb_ood = ""
    if len(X_ood) > 0:
        y_pred_nb_ood = pipeline_nb.predict(X_ood)
        acc_nb_ood = accuracy_score(y_ood, y_pred_nb_ood)
        report_nb_ood = classification_report(y_ood, y_pred_nb_ood,
                                              target_names=sorted(set(y_all)), digits=4)

    if verbose:
        print(f"  Accuracy (test IID): {acc_nb:.4f}")
        if len(X_ood) > 0:
            print(f"  Accuracy (test OOD): {acc_nb_ood:.4f}")
        print(f"  F1 macro (CV):       {cv_nb.mean():.4f} ± {cv_nb.std():.4f}")

    # ── SVM ───────────────────────────────────────────────────────────────────
    if verbose:
        print("\n── Entrenando SVM ──────────────────────────────────────────────")

    if usar_grid_search:
        gs_svm = GridSearchCV(construir_pipeline_svm(), GRID_SVM,
                              cv=cv_gs, scoring="f1_macro", n_jobs=-1, verbose=0)
        gs_svm.fit(X_train, y_train)
        pipeline_svm = gs_svm.best_estimator_
        if verbose:
            print(f"  Mejores params SVM: {gs_svm.best_params_}")
            print(f"  Mejor F1 (CV):      {gs_svm.best_score_:.4f}")
    else:
        pipeline_svm = construir_pipeline_svm()
        pipeline_svm.fit(X_train, y_train)

    cv_svm     = cross_val_score(pipeline_svm, X_train, y_train,
                                 cv=cv_for_scores, scoring="f1_macro", n_jobs=-1)
    y_pred_svm = pipeline_svm.predict(X_test)
    report_svm = classification_report(y_test, y_pred_svm,
                                       target_names=sorted(set(y_all)), digits=4)
    acc_svm    = accuracy_score(y_test, y_pred_svm)

    y_pred_svm_ood = np.array([], dtype=str)
    acc_svm_ood = np.nan
    report_svm_ood = ""
    if len(X_ood) > 0:
        y_pred_svm_ood = pipeline_svm.predict(X_ood)
        acc_svm_ood = accuracy_score(y_ood, y_pred_svm_ood)
        report_svm_ood = classification_report(y_ood, y_pred_svm_ood,
                                               target_names=sorted(set(y_all)), digits=4)

    if verbose:
        print(f"  Accuracy (test IID): {acc_svm:.4f}")
        if len(X_ood) > 0:
            print(f"  Accuracy (test OOD): {acc_svm_ood:.4f}")
        print(f"  F1 macro (CV):       {cv_svm.mean():.4f} ± {cv_svm.std():.4f}")

    # ── Guardar modelos ───────────────────────────────────────────────────────
    joblib.dump(pipeline_nb,  NB_MODEL_PKL)
    joblib.dump(pipeline_svm, SVM_MODEL_PKL)
    if verbose:
        print(f"\n✓ Modelos guardados → {MODELS_DIR}")

    return {
        "X_train":      X_train,
        "X_test":       X_test,
        "X_ood":        X_ood,
        "y_train":      y_train,
        "y_test":       y_test,
        "y_ood":        y_ood,
        "pipeline_nb":  pipeline_nb,
        "pipeline_svm": pipeline_svm,
        "y_pred_nb":    y_pred_nb,
        "y_pred_svm":   y_pred_svm,
        "y_pred_nb_ood": y_pred_nb_ood,
        "y_pred_svm_ood": y_pred_svm_ood,
        "report_nb":    report_nb,
        "report_svm":   report_svm,
        "report_nb_ood": report_nb_ood,
        "report_svm_ood": report_svm_ood,
        "cv_nb":        cv_nb,
        "cv_svm":       cv_svm,
        "acc_nb":       acc_nb,
        "acc_svm":      acc_svm,
        "acc_nb_ood":   acc_nb_ood,
        "acc_svm_ood":  acc_svm_ood,
    }


if __name__ == "__main__":
    from src.preprocessor import preprocesar_corpus
    df = pd.read_csv(DATASET_CSV)
    df = preprocesar_corpus(df)
    resultados = entrenar(df, usar_grid_search=True)
    print("\n── Reporte Naive Bayes ─────────────────────────────────────────")
    print(resultados["report_nb"])
    print("\n── Reporte SVM ─────────────────────────────────────────────────")
    print(resultados["report_svm"])
