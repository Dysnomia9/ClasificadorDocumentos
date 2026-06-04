"""
Módulo de análisis de características TF-IDF.

Genera visualizaciones explicativas del modelo:
  - Top N palabras más discriminativas por clase (barras horizontales)
  - Mapa de calor: peso TF-IDF promedio por clase vs top palabras globales
  - Tabla de términos exclusivos por clase

Sirven para interpretar qué términos pesan más en cada clase.
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer

from config import REPORTS_DIR, MODELS_DIR, DATASET_CSV, SVM_MODEL_PKL, CLASES

PALETTE = {
    "factura":  "#2980b9",
    "contrato": "#e74c3c",
    "informe":  "#27ae60",
    "correo":   "#8e44ad",
}

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})


def _obtener_top_palabras_por_clase(pipeline, X_texts: np.ndarray,
                                     y_labels: np.ndarray,
                                     n_top: int = 20) -> dict:
    """
    Calcula el peso TF-IDF promedio de cada término por clase.
    Retorna {clase: [(palabra, peso_promedio), ...]} con los n_top términos.
    """
    vectorizer: TfidfVectorizer = pipeline.named_steps["tfidf"]
    nombres    = np.array(vectorizer.get_feature_names_out())
    X_tfidf    = vectorizer.transform(X_texts)

    top_por_clase = {}
    for clase in sorted(set(y_labels)):
        mask         = (y_labels == clase)
        X_clase      = X_tfidf[mask]
        medias        = np.asarray(X_clase.mean(axis=0)).flatten()
        indices_top   = medias.argsort()[::-1][:n_top]
        top_por_clase[clase] = [(nombres[i], medias[i]) for i in indices_top]

    return top_por_clase


def graficar_top_palabras(pipeline, X_texts: np.ndarray,
                           y_labels: np.ndarray, n_top: int = 15) -> Path:
    """
    Gráfico de barras horizontales: top N palabras TF-IDF por clase.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    top = _obtener_top_palabras_por_clase(pipeline, X_texts, y_labels, n_top)

    clases_presentes = sorted(top.keys())
    ncols = 2
    nrows = (len(clases_presentes) + 1) // 2

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(14, nrows * 4 + 1),
                              constrained_layout=True)
    axes = axes.flatten()

    for ax, clase in zip(axes, clases_presentes):
        palabras, pesos = zip(*top[clase])
        color = PALETTE.get(clase, "#555")
        ax.barh(palabras[::-1], pesos[::-1], color=color, alpha=0.85, edgecolor="white")
        ax.set_title(f"Top {n_top} palabras — {clase.upper()}", fontsize=11,
                      fontweight="bold", color=color)
        ax.set_xlabel("Peso TF-IDF promedio", fontsize=9)
        ax.tick_params(axis="y", labelsize=8)

    # Ocultar ejes sobrantes
    for ax in axes[len(clases_presentes):]:
        ax.set_visible(False)

    fig.suptitle("Términos más discriminativos por clase (TF-IDF)",
                  fontsize=14, fontweight="bold")

    ruta = REPORTS_DIR / "tfidf_top_palabras.png"
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    return ruta


def graficar_heatmap_tfidf(pipeline, X_texts: np.ndarray,
                            y_labels: np.ndarray, n_top: int = 20) -> Path:
    """
    Mapa de calor: clases × top palabras globales, con peso TF-IDF promedio.
    Permite ver qué palabras son exclusivas vs compartidas entre clases.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    vectorizer: TfidfVectorizer = pipeline.named_steps["tfidf"]
    nombres    = np.array(vectorizer.get_feature_names_out())
    X_tfidf    = vectorizer.transform(X_texts)

    clases_presentes = sorted(set(y_labels))

    # Calcular medias por clase para TODAS las palabras
    medias_por_clase = {}
    for clase in clases_presentes:
        mask = (y_labels == clase)
        medias_por_clase[clase] = np.asarray(X_tfidf[mask].mean(axis=0)).flatten()

    # Top N palabras según varianza entre clases (máxima discriminación)
    matriz_clases = np.vstack([medias_por_clase[c] for c in clases_presentes])
    varianza      = matriz_clases.var(axis=0)
    top_indices   = varianza.argsort()[::-1][:n_top]
    top_palabras  = nombres[top_indices]

    # Construir DataFrame para el heatmap
    df_heat = pd.DataFrame(
        {clase: medias_por_clase[clase][top_indices] for clase in clases_presentes},
        index=top_palabras,
    ).T  # clases en filas, palabras en columnas

    fig, ax = plt.subplots(figsize=(max(12, n_top * 0.65), len(clases_presentes) * 1.2 + 2))
    sns.heatmap(
        df_heat, annot=True, fmt=".3f", cmap="YlOrRd",
        linewidths=0.4, ax=ax,
        cbar_kws={"label": "Peso TF-IDF promedio", "shrink": 0.6},
        annot_kws={"size": 7},
    )
    ax.set_title("Mapa de calor TF-IDF: Términos discriminativos por clase",
                  fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Término", fontsize=10)
    ax.set_ylabel("Clase", fontsize=10)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)

    plt.tight_layout()
    ruta = REPORTS_DIR / "tfidf_heatmap.png"
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    return ruta


def generar_tabla_terminos_exclusivos(pipeline, X_texts: np.ndarray,
                                       y_labels: np.ndarray,
                                       n_exclusivos: int = 10) -> pd.DataFrame:
    """
    Genera una tabla con los términos más exclusivos de cada clase:
    palabras con alto peso en la clase y bajo peso en las demás.
    Guarda como CSV en reports/.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    vectorizer: TfidfVectorizer = pipeline.named_steps["tfidf"]
    nombres    = np.array(vectorizer.get_feature_names_out())
    X_tfidf    = vectorizer.transform(X_texts)

    clases_presentes = sorted(set(y_labels))
    medias_por_clase = {
        c: np.asarray(X_tfidf[y_labels == c].mean(axis=0)).flatten()
        for c in clases_presentes
    }

    filas = []
    for clase in clases_presentes:
        media_propia  = medias_por_clase[clase]
        media_resto   = np.mean(
            [medias_por_clase[c] for c in clases_presentes if c != clase], axis=0
        )
        # Score de exclusividad: cuánto más pesa en esta clase vs las demás
        exclusividad  = media_propia - media_resto
        top_idx       = exclusividad.argsort()[::-1][:n_exclusivos]
        for rank, idx in enumerate(top_idx, 1):
            filas.append({
                "clase":          clase,
                "rank":           rank,
                "término":        nombres[idx],
                "peso_propio":    round(float(media_propia[idx]),  4),
                "peso_resto":     round(float(media_resto[idx]),   4),
                "exclusividad":   round(float(exclusividad[idx]),  4),
            })

    df_excl = pd.DataFrame(filas)
    ruta_csv = REPORTS_DIR / "terminos_exclusivos.csv"
    df_excl.to_csv(ruta_csv, index=False)
    return df_excl


def analizar_features(verbose: bool = True) -> dict:
    """
    Ejecuta todo el análisis de características.
    Carga el modelo SVM entrenado y el dataset procesado.

    Retorna dict con rutas a los archivos generados.
    """
    if not SVM_MODEL_PKL.exists():
        raise FileNotFoundError("Modelo SVM no encontrado. Ejecuta: python main.py train")

    from src.preprocessor import preprocesar_corpus

    pipeline = joblib.load(SVM_MODEL_PKL)
    df       = pd.read_csv(DATASET_CSV)

    if "texto_limpio" not in df.columns or df["texto_limpio"].isna().any():
        df = preprocesar_corpus(df)

    X = df["texto_limpio"].values
    y = df["clase"].values

    archivos = {}

    if verbose:
        print("  → Graficando top palabras por clase...")
    archivos["top_palabras"] = graficar_top_palabras(pipeline, X, y, n_top=15)

    if verbose:
        print("  → Generando heatmap TF-IDF...")
    archivos["heatmap"] = graficar_heatmap_tfidf(pipeline, X, y, n_top=20)

    if verbose:
        print("  → Calculando términos exclusivos por clase...")
    df_excl = generar_tabla_terminos_exclusivos(pipeline, X, y)
    archivos["exclusivos_csv"] = REPORTS_DIR / "terminos_exclusivos.csv"

    if verbose:
        print(f"\n✓ Análisis de features guardado en: {REPORTS_DIR}")
        for k, v in archivos.items():
            print(f"   {k}: {v}")

        print("\nTop 5 términos exclusivos por clase:")
        print("─" * 50)
        for clase in sorted(df_excl["clase"].unique()):
            terminos = df_excl[df_excl["clase"] == clase].head(5)["término"].tolist()
            print(f"  {clase:12s}: {', '.join(terminos)}")

    return archivos


if __name__ == "__main__":
    analizar_features(verbose=True)
