"""
Módulo de evaluación y visualización de resultados.

Genera:
  - Matrices de confusión (NB y SVM)
  - Curvas de aprendizaje (NB vs SVM)
  - Gráfico de comparación de métricas
  - Reporte de texto completo
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")    # sin GUI
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.model_selection import learning_curve, StratifiedKFold

from config import REPORTS_DIR, CLASES, RANDOM_STATE

# Estilo visual consistente
plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi": 150,
})
PALETTE = ["#2c3e50", "#e74c3c", "#27ae60", "#2980b9"]


# ─── Matriz de confusión ─────────────────────────────────────────────────────

def graficar_confusion(y_true, y_pred, nombre_modelo: str,
                       clases: list = CLASES, split_tag: str = "iid") -> Path:
    """Genera y guarda la matriz de confusión normalizada."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cm  = confusion_matrix(y_true, y_pred, labels=clases, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm * 100, annot=True, fmt=".1f", cmap="Blues",
        xticklabels=clases, yticklabels=clases,
        linewidths=0.5, ax=ax, cbar_kws={"label": "% predicciones"},
    )
    ax.set_xlabel("Predicción", fontsize=11)
    ax.set_ylabel("Clase real", fontsize=11)
    ax.set_title(f"Matriz de Confusión {split_tag.upper()} — {nombre_modelo}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    ruta = REPORTS_DIR / f"confusion_{nombre_modelo.lower().replace(' ','_')}_{split_tag.lower()}.png"
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    return ruta


# ─── Curvas de aprendizaje ────────────────────────────────────────────────────

def graficar_curvas_aprendizaje(pipelines: dict,
                                 X_train, y_train) -> Path:
    """
    Genera curvas de aprendizaje para cada modelo.
    pipelines: {'Naive Bayes': pipeline_nb, 'SVM': pipeline_svm}
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    sizes = np.linspace(0.30, 1.0, 6)

    fig, axes = plt.subplots(1, len(pipelines), figsize=(6 * len(pipelines), 5), sharey=True)
    if len(pipelines) == 1:
        axes = [axes]

    for ax, (nombre, pipeline), color in zip(axes, pipelines.items(), PALETTE):
        train_sizes, train_scores, val_scores = learning_curve(
            pipeline, X_train, y_train,
            train_sizes=sizes, cv=cv,
            scoring="f1_macro", n_jobs=-1, error_score=np.nan,
        )

        valid_mask = np.isfinite(train_scores).all(axis=1) & np.isfinite(val_scores).all(axis=1)
        train_sizes = train_sizes[valid_mask]
        train_scores = train_scores[valid_mask]
        val_scores = val_scores[valid_mask]
        if len(train_sizes) == 0:
            continue

        tr_mean, tr_std = train_scores.mean(1), train_scores.std(1)
        vl_mean, vl_std = val_scores.mean(1),   val_scores.std(1)

        ax.plot(train_sizes, tr_mean, "o-", color=color, label="Train", lw=2)
        ax.fill_between(train_sizes, tr_mean - tr_std, tr_mean + tr_std,
                         alpha=0.15, color=color)
        ax.plot(train_sizes, vl_mean, "s--", color=color, label="Validación", lw=2, alpha=0.8)
        ax.fill_between(train_sizes, vl_mean - vl_std, vl_mean + vl_std,
                         alpha=0.10, color=color)

        ax.set_title(f"Curva de Aprendizaje — {nombre}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Tamaño del conjunto de entrenamiento")
        ax.set_ylabel("F1 macro")
        ax.legend(loc="lower right")
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

    plt.suptitle("Curvas de Aprendizaje", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    ruta = REPORTS_DIR / "curvas_aprendizaje.png"
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    return ruta


# ─── Comparación de métricas ─────────────────────────────────────────────────

def graficar_comparacion_metricas(resultados: dict) -> dict:
    """Bar chart comparando accuracy, precision, recall y F1 de ambos modelos.
    Genera gráficos separados para datos IID y OOD cuando existan."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    y_test     = resultados["y_test"]
    y_pred_nb  = resultados["y_pred_nb"]
    y_pred_svm = resultados["y_pred_svm"]

    y_ood = resultados.get("y_ood", np.array([], dtype=str))
    y_pred_nb_ood = resultados.get("y_pred_nb_ood", np.array([], dtype=str))
    y_pred_svm_ood = resultados.get("y_pred_svm_ood", np.array([], dtype=str))

    metricas_iid = {
        "Accuracy":  [accuracy_score(y_test, y_pred_nb),  accuracy_score(y_test, y_pred_svm)],
        "Precision": [precision_score(y_test, y_pred_nb,  average="macro", zero_division=0),
                      precision_score(y_test, y_pred_svm, average="macro", zero_division=0)],
        "Recall":    [recall_score(y_test, y_pred_nb,  average="macro", zero_division=0),
                      recall_score(y_test, y_pred_svm, average="macro", zero_division=0)],
        "F1 Macro":  [f1_score(y_test, y_pred_nb,  average="macro"),
                      f1_score(y_test, y_pred_svm, average="macro")],
    }

    rutas = {}
    modelos = ["Naive Bayes", "SVM"]
    colores = [PALETTE[0], PALETTE[1]]

    def _guardar_barras(metricas: dict, titulo: str, nombre_archivo: str) -> Path:
        """Función auxiliar para generar y guardar gráfico de barras."""
        x = np.arange(len(metricas))
        width = 0.35
        fig, ax = plt.subplots(figsize=(9, 5))
        for i, (modelo, color) in enumerate(zip(modelos, colores)):
            vals = [metricas[m][i] for m in metricas]
            bars = ax.bar(x + i * width - width/2, vals, width, label=modelo,
                          color=color, alpha=0.85, edgecolor="white")
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(list(metricas.keys()), fontsize=11)
        ax.set_ylim(0, 1.12)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
        ax.set_ylabel("Valor de la métrica", fontsize=11)
        ax.set_title(titulo, fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        plt.tight_layout()

        ruta_local = REPORTS_DIR / nombre_archivo
        fig.savefig(ruta_local, bbox_inches="tight")
        plt.close(fig)
        return ruta_local

    rutas["comparacion_iid"] = _guardar_barras(
        metricas_iid,
        "Comparación de Métricas IID: Naive Bayes vs SVM",
        "comparacion_metricas_iid.png",
    )

    if len(y_ood) > 0 and len(y_pred_nb_ood) == len(y_ood) and len(y_pred_svm_ood) == len(y_ood):
        metricas_ood = {
            "Accuracy":  [accuracy_score(y_ood, y_pred_nb_ood),  accuracy_score(y_ood, y_pred_svm_ood)],
            "Precision": [precision_score(y_ood, y_pred_nb_ood,  average="macro", zero_division=0),
                          precision_score(y_ood, y_pred_svm_ood, average="macro", zero_division=0)],
            "Recall":    [recall_score(y_ood, y_pred_nb_ood,  average="macro", zero_division=0),
                          recall_score(y_ood, y_pred_svm_ood, average="macro", zero_division=0)],
            "F1 Macro":  [f1_score(y_ood, y_pred_nb_ood,  average="macro"),
                          f1_score(y_ood, y_pred_svm_ood, average="macro")],
        }
        rutas["comparacion_ood"] = _guardar_barras(
            metricas_ood,
            "Comparación de Métricas OOD: Naive Bayes vs SVM",
            "comparacion_metricas_ood.png",
        )

    return rutas


# ─── Reporte de texto ─────────────────────────────────────────────────────────

def generar_reporte_texto(resultados: dict) -> str:
    """Genera y guarda el reporte completo en texto plano."""
    from config import REPORT_TXT
    import numpy as np

    y_test, y_pred_nb, y_pred_svm = (
        resultados["y_test"], resultados["y_pred_nb"], resultados["y_pred_svm"]
    )
    cv_nb, cv_svm = resultados["cv_nb"], resultados["cv_svm"]
    y_ood = resultados.get("y_ood", np.array([], dtype=str))
    y_pred_nb_ood = resultados.get("y_pred_nb_ood", np.array([], dtype=str))
    y_pred_svm_ood = resultados.get("y_pred_svm_ood", np.array([], dtype=str))

    lineas = [
        "=" * 65,
        "   REPORTE DE EVALUACIÓN — CLASIFICADOR DE DOCUMENTOS",
        "=" * 65,
        "",
        "CONFIGURACIÓN",
        f"  Clases:              {', '.join(sorted(set(y_test)))}",
        f"  Documentos de test:  {len(y_test)}",
    ]
    if len(y_ood) > 0:
        lineas.append(f"  Documentos test OOD: {len(y_ood)}")

    lineas += [
        "",
        "─" * 65,
        "NAIVE BAYES",
        "─" * 65,
        f"  Accuracy (test):  {accuracy_score(y_test, y_pred_nb):.4f}",
        f"  F1 macro  (test):  {f1_score(y_test, y_pred_nb, average='macro'):.4f}",
        f"  F1 macro  (CV 5-fold): {cv_nb.mean():.4f} ± {cv_nb.std():.4f}",
        "",
        "  Reporte por clase:",
        resultados["report_nb"],
        "",
    ]

    if len(y_ood) > 0 and len(y_pred_nb_ood) == len(y_ood):
        lineas += [
            "  Test OOD:",
            f"    Accuracy OOD: {accuracy_score(y_ood, y_pred_nb_ood):.4f}",
            f"    F1 macro OOD: {f1_score(y_ood, y_pred_nb_ood, average='macro'):.4f}",
            resultados.get("report_nb_ood", ""),
            "",
        ]

    lineas += [
        "─" * 65,
        "SVM (Linear SVC)",
        "─" * 65,
        f"  Accuracy (test):  {accuracy_score(y_test, y_pred_svm):.4f}",
        f"  F1 macro  (test):  {f1_score(y_test, y_pred_svm, average='macro'):.4f}",
        f"  F1 macro  (CV 5-fold): {cv_svm.mean():.4f} ± {cv_svm.std():.4f}",
        "",
        "  Reporte por clase:",
        resultados["report_svm"],
        "",
    ]

    if len(y_ood) > 0 and len(y_pred_svm_ood) == len(y_ood):
        lineas += [
            "  Test OOD:",
            f"    Accuracy OOD: {accuracy_score(y_ood, y_pred_svm_ood):.4f}",
            f"    F1 macro OOD: {f1_score(y_ood, y_pred_svm_ood, average='macro'):.4f}",
            resultados.get("report_svm_ood", ""),
            "",
        ]

    lineas += [
        "─" * 65,
        "CONCLUSIÓN COMPARATIVA",
        "─" * 65,
    ]

    nb_f1  = f1_score(y_test, y_pred_nb,  average="macro")
    svm_f1 = f1_score(y_test, y_pred_svm, average="macro")
    diff    = abs(svm_f1 - nb_f1)
    if diff < 1e-9:
        ganador = "Empate"
    else:
        ganador = "SVM" if svm_f1 > nb_f1 else "Naive Bayes"

    lineas += [
        f"  Mejor modelo (F1 macro IID): {ganador} ({diff:.4f} pts de diferencia)",
        f"  SVM F1 IID:          {svm_f1:.4f}",
        f"  Naive Bayes F1 IID:  {nb_f1:.4f}",
    ]

    if len(y_ood) > 0 and len(y_pred_nb_ood) == len(y_ood) and len(y_pred_svm_ood) == len(y_ood):
        nb_f1_ood = f1_score(y_ood, y_pred_nb_ood, average="macro")
        svm_f1_ood = f1_score(y_ood, y_pred_svm_ood, average="macro")
        diff_ood = abs(svm_f1_ood - nb_f1_ood)
        if diff_ood < 1e-9:
            ganador_ood = "Empate"
        else:
            ganador_ood = "SVM" if svm_f1_ood > nb_f1_ood else "Naive Bayes"
        lineas += [
            "",
            f"  Mejor modelo (F1 macro OOD): {ganador_ood} ({diff_ood:.4f} pts de diferencia)",
            f"  SVM F1 OOD:          {svm_f1_ood:.4f}",
            f"  Naive Bayes F1 OOD:  {nb_f1_ood:.4f}",
        ]

    lineas += [
        "",
        "  Nota: LinearSVC es generalmente superior en clasificación de texto",
        "  con TF-IDF por su margen de separación óptimo. Naive Bayes es más",
        "  rápido y explicable, adecuado para datasets pequeños.",
        "=" * 65,
    ]

    texto = "\n".join(lineas)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_TXT.write_text(texto, encoding="utf-8")
    return texto


# ─── Función de evaluación completa ──────────────────────────────────────────

def evaluar_completo(resultados: dict, verbose: bool = True) -> dict:
    """
    Ejecuta todas las visualizaciones y el reporte de texto.
    Retorna diccionario con rutas a los archivos generados.
    """
    archivos = {}

    if verbose:
        print("  → Generando matrices de confusión...")
    archivos["conf_nb_iid"] = graficar_confusion(
        resultados["y_test"], resultados["y_pred_nb"], "Naive Bayes", split_tag="iid")
    archivos["conf_svm_iid"] = graficar_confusion(
        resultados["y_test"], resultados["y_pred_svm"], "SVM", split_tag="iid")

    y_ood = resultados.get("y_ood", np.array([], dtype=str))
    y_pred_nb_ood = resultados.get("y_pred_nb_ood", np.array([], dtype=str))
    y_pred_svm_ood = resultados.get("y_pred_svm_ood", np.array([], dtype=str))
    if len(y_ood) > 0 and len(y_pred_nb_ood) == len(y_ood) and len(y_pred_svm_ood) == len(y_ood):
        archivos["conf_nb_ood"] = graficar_confusion(
            y_ood, y_pred_nb_ood, "Naive Bayes", split_tag="ood")
        archivos["conf_svm_ood"] = graficar_confusion(
            y_ood, y_pred_svm_ood, "SVM", split_tag="ood")

    if verbose:
        print("  → Generando curvas de aprendizaje (tarda ~30s)...")
    archivos["curvas"] = graficar_curvas_aprendizaje(
        {"Naive Bayes": resultados["pipeline_nb"], "SVM": resultados["pipeline_svm"]},
        resultados["X_train"], resultados["y_train"],
    )

    if verbose:
        print("  → Generando gráficos de comparación IID/OOD...")
    rutas_comparacion = graficar_comparacion_metricas(resultados)
    archivos.update(rutas_comparacion)

    if verbose:
        print("  → Generando reporte de texto...")
    generar_reporte_texto(resultados)
    archivos["reporte_txt"] = str(REPORTS_DIR / "reporte_comparacion.txt")

    if verbose:
        print(f"\n✓ Reportes guardados en: {REPORTS_DIR}")
        for k, v in archivos.items():
            print(f"   {k}: {v}")

    return archivos


if __name__ == "__main__":
    print("Ejecuta main.py evaluate para generar reportes completos.")
