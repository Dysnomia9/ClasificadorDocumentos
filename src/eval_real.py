"""
Evaluación con Documentos Reales
=================================
Genera figuras y métricas evaluando sobre PDFs reales
proporcionados por el usuario, en lugar del corpus sintético.

Uso rápido — carpeta organizada por clase:
  Crea la siguiente estructura y ejecuta:

    docs_reales/
      factura/   ← pon aquí tus PDFs de facturas
      contrato/  ← tus contratos
      informe/   ← tus informes
      correo/    ← tus correos

  python eval_real.py --input docs_reales/

Uso individual — pasar PDFs sueltos con etiqueta:
  python eval_real.py \\
      --pdf mi_factura.pdf factura \\
      --pdf mi_contrato.pdf contrato \\
      --pdf otro_informe.pdf informe

Los resultados se guardan en reports/evaluacion_real/
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, f1_score, precision_score, recall_score,
)

# ─── Rutas ────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
REPORTS_REAL = BASE_DIR / "reports" / "evaluacion_real"
CLASES       = ["factura", "contrato", "informe", "cotizacion", "orden_compra"]

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})
PALETTE = ["#2c3e50", "#e74c3c", "#27ae60", "#2980b9"]
COLORES_CLASE = {
    "factura":      "#2980b9",
    "contrato":     "#e74c3c",
    "informe":      "#27ae60",
    "cotizacion":   "#f39c12",
    "orden_compra": "#16a085",
}


# ─── Carga de modelos ─────────────────────────────────────────────────────────

def _cargar_modelos():
    from config import NB_MODEL_PKL, SVM_MODEL_PKL
    if not NB_MODEL_PKL.exists() or not SVM_MODEL_PKL.exists():
        print("[ERROR] Modelos no encontrados.")
        print("  Ejecuta primero: python main.py realistic && python main.py extract && python main.py train")
        sys.exit(1)
    nb  = joblib.load(NB_MODEL_PKL)
    svm = joblib.load(SVM_MODEL_PKL)
    print(f"✓ Modelos cargados desde {NB_MODEL_PKL.parent}")
    return nb, svm


# ─── Extracción y preprocesamiento ────────────────────────────────────────────

def _procesar_pdf(pdf_path: Path) -> str:
    """Extrae y limpia el texto de un PDF."""
    import fitz
    from src.preprocessor import limpiar_texto

    try:
        doc = fitz.open(str(pdf_path))
        texto = "".join(p.get_text("text") for p in doc)
        doc.close()
    except Exception as e:
        print(f"  [WARN] No se pudo leer {pdf_path.name}: {e}")
        return ""

    # Fallback OCR básico si el texto es muy corto
    if len(texto.strip()) < 50:
        try:
            import pytesseract
            from PIL import Image
            import io
            doc = fitz.open(str(pdf_path))
            texto = ""
            for pag in doc:
                pix = pag.get_pixmap(matrix=fitz.Matrix(2, 2))
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                texto += pytesseract.image_to_string(img, lang="spa")
            doc.close()
        except Exception:
            pass

    return limpiar_texto(texto)


def _predecir_uno(texto_limpio: str, pipeline_nb, pipeline_svm) -> dict:
    """Predice clase y confianza para un documento."""
    resultado = {}
    for nombre, pipeline in [("nb", pipeline_nb), ("svm", pipeline_svm)]:
        clase_pred = pipeline.predict([texto_limpio])[0]
        clf   = pipeline.named_steps["clf"]
        tfidf = pipeline.named_steps["tfidf"]
        X_vec = tfidf.transform([texto_limpio])
        if hasattr(clf, "decision_function"):
            scores = clf.decision_function(X_vec)[0]
            e = np.exp(scores - scores.max())
            probs = e / e.sum()
        else:
            probs = clf.predict_proba(X_vec)[0]
        resultado[nombre] = {
            "clase":     clase_pred,
            "confianza": float(probs.max()),
            "scores":    {c: float(p) for c, p in zip(pipeline.classes_, probs)},
        }
    return resultado


# ─── Recolección de documentos ────────────────────────────────────────────────

def _recolectar_desde_carpeta(carpeta: Path) -> list[tuple[Path, str]]:
    """
    Busca PDFs en carpeta/<clase>/*.pdf
    Retorna lista de (ruta_pdf, clase_real).
    """
    docs = []
    for clase in CLASES:
        subdir = carpeta / clase
        if not subdir.exists():
            print(f"  [WARN] Carpeta no encontrada: {subdir}")
            continue
        pdfs = sorted(subdir.glob("*.pdf"))
        if not pdfs:
            print(f"  [WARN] Sin PDFs en {subdir}")
            continue
        for pdf in pdfs:
            docs.append((pdf, clase))
        print(f"  ✓ {clase:12s}: {len(pdfs)} PDF(s)")
    return docs


# ─── Evaluación ───────────────────────────────────────────────────────────────

def evaluar(docs: list[tuple[Path, str]],
            pipeline_nb, pipeline_svm,
            verbose: bool = True) -> pd.DataFrame:
    """
    Procesa todos los documentos y devuelve DataFrame con predicciones.
    """
    filas = []
    n = len(docs)

    print(f"\n  Procesando {n} documento(s)...")
    for i, (pdf_path, clase_real) in enumerate(docs):
        texto = _procesar_pdf(pdf_path)
        n_tokens = len(texto.split())

        if not texto.strip():
            print(f"  [SKIP] {pdf_path.name} — sin texto extraíble")
            continue

        pred = _predecir_uno(texto, pipeline_nb, pipeline_svm)

        filas.append({
            "archivo":       pdf_path.name,
            "clase_real":    clase_real,
            "clase_nb":      pred["nb"]["clase"],
            "conf_nb":       pred["nb"]["confianza"],
            "clase_svm":     pred["svm"]["clase"],
            "conf_svm":      pred["svm"]["confianza"],
            "tokens":        n_tokens,
            "coinciden":     pred["nb"]["clase"] == pred["svm"]["clase"],
            "nb_ok":         pred["nb"]["clase"] == clase_real,
            "svm_ok":        pred["svm"]["clase"] == clase_real,
        })

        if verbose:
            nb_ok  = "✓" if filas[-1]["nb_ok"]  else "✗"
            svm_ok = "✓" if filas[-1]["svm_ok"] else "✗"
            print(f"  [{i+1:>3}/{n}] {pdf_path.name[:40]:<40} "
                  f"real={clase_real:<10} "
                  f"NB={pred['nb']['clase']:<10}{nb_ok}  "
                  f"SVM={pred['svm']['clase']:<10}{svm_ok}")

    return pd.DataFrame(filas)


# ─── Figuras ──────────────────────────────────────────────────────────────────

def _figura_confusion(y_true, y_pred, nombre_modelo: str, ruta: Path) -> None:
    clases_presentes = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred,
                          labels=clases_presentes, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm * 100, annot=True, fmt=".1f", cmap="Blues",
        xticklabels=clases_presentes, yticklabels=clases_presentes,
        linewidths=0.5, ax=ax, cbar_kws={"label": "% predicciones"},
        vmin=0, vmax=100,
    )
    ax.set_xlabel("Predicción", fontsize=11)
    ax.set_ylabel("Clase real",  fontsize=11)
    ax.set_title(f"Matriz de Confusión — {nombre_modelo}\n(Documentos reales)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {ruta.name}")


def _figura_metricas(df: pd.DataFrame, ruta: Path) -> None:
    y_true = df["clase_real"].values
    y_nb   = df["clase_nb"].values
    y_svm  = df["clase_svm"].values

    metricas = {
        "Accuracy":  [accuracy_score(y_true, y_nb),
                      accuracy_score(y_true, y_svm)],
        "Precision": [precision_score(y_true, y_nb,  average="macro", zero_division=0),
                      precision_score(y_true, y_svm, average="macro", zero_division=0)],
        "Recall":    [recall_score(y_true, y_nb,  average="macro", zero_division=0),
                      recall_score(y_true, y_svm, average="macro", zero_division=0)],
        "F1 Macro":  [f1_score(y_true, y_nb,  average="macro", zero_division=0),
                      f1_score(y_true, y_svm, average="macro", zero_division=0)],
    }

    x      = np.arange(len(metricas))
    width  = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (modelo, color) in enumerate(
        [("Naive Bayes", PALETTE[0]), ("SVM Lineal", PALETTE[1])]
    ):
        vals = [metricas[m][i] for m in metricas]
        bars = ax.bar(x + i * width - width / 2, vals, width,
                      label=modelo, color=color, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(list(metricas.keys()), fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.set_ylabel("Valor de la métrica", fontsize=11)
    ax.set_title("Comparación de Métricas — Documentos Reales\nNaive Bayes vs SVM Lineal",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {ruta.name}")


def _figura_por_clase(df: pd.DataFrame, ruta: Path) -> None:
    """F1 por clase para cada modelo — barras agrupadas."""
    y_true = df["clase_real"].values
    clases = sorted(df["clase_real"].unique())

    f1_nb  = {c: f1_score(y_true == c, df["clase_nb"].values  == c, zero_division=0)
               for c in clases}
    f1_svm = {c: f1_score(y_true == c, df["clase_svm"].values == c, zero_division=0)
               for c in clases}

    x     = np.arange(len(clases))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))

    bars_nb  = ax.bar(x - width/2, [f1_nb[c]  for c in clases], width,
                      label="Naive Bayes", color=PALETTE[0], alpha=0.85, edgecolor="white")
    bars_svm = ax.bar(x + width/2, [f1_svm[c] for c in clases], width,
                      label="SVM Lineal",  color=PALETTE[1], alpha=0.85, edgecolor="white")

    for bars in [bars_nb, bars_svm]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([c.upper() for c in clases], fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_title("F1 Score por Clase — Documentos Reales",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {ruta.name}")


def _figura_confianza(df: pd.DataFrame, ruta: Path) -> None:
    """Boxplot de confianza del SVM por clase, coloreado por si acertó o no."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for ax, (modelo, col_clase, col_conf, col_ok) in zip(axes, [
        ("Naive Bayes", "clase_svm", "conf_nb",  "nb_ok"),
        ("SVM Lineal",  "clase_svm", "conf_svm", "svm_ok"),
    ]):
        for i, clase in enumerate(CLASES):
            sub = df[df["clase_real"] == clase]
            if sub.empty:
                continue
            conf_ok  = sub.loc[sub[col_ok],  col_conf].values
            conf_err = sub.loc[~sub[col_ok], col_conf].values
            color = COLORES_CLASE[clase]
            if len(conf_ok):
                ax.scatter([i] * len(conf_ok),  conf_ok,
                           color=color, alpha=0.75, s=60, zorder=3, label=None)
            if len(conf_err):
                ax.scatter([i] * len(conf_err), conf_err,
                           color=color, alpha=0.75, s=80, marker="X",
                           edgecolors="black", linewidths=0.5, zorder=4,
                           label=None)

        ax.set_xticks(range(len(CLASES)))
        ax.set_xticklabels([c.upper() for c in CLASES], fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
        ax.set_ylabel("Confianza del modelo", fontsize=9)
        ax.set_title(modelo, fontsize=11, fontweight="bold")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.text(len(CLASES)-0.5, 0.51, "umbral 50%", fontsize=7, color="gray")

    # Leyenda manual: ● = correcto, ✕ = error
    from matplotlib.lines import Line2D
    leyenda = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#555", markersize=7,
               label="Predicción correcta"),
        Line2D([0],[0], marker="X", color="w", markerfacecolor="#555", markersize=7,
               markeredgecolor="black", markeredgewidth=0.5, label="Error de clasificación"),
    ]
    fig.legend(handles=leyenda, loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Distribución de Confianza por Clase — Documentos Reales",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {ruta.name}")


# ─── Reporte de texto ─────────────────────────────────────────────────────────

def _reporte_texto(df: pd.DataFrame, ruta: Path) -> None:
    y_true = df["clase_real"].values
    y_nb   = df["clase_nb"].values
    y_svm  = df["clase_svm"].values
    clases = sorted(df["clase_real"].unique())

    acc_nb  = accuracy_score(y_true, y_nb)
    acc_svm = accuracy_score(y_true, y_svm)
    f1_nb   = f1_score(y_true, y_nb,  average="macro", zero_division=0)
    f1_svm  = f1_score(y_true, y_svm, average="macro", zero_division=0)

    lineas = [
        "=" * 65,
        "   EVALUACIÓN CON DOCUMENTOS REALES — CLASIFICADOR",
        "=" * 65,
        "",
        f"  Total documentos evaluados : {len(df)}",
        f"  Clases presentes           : {', '.join(clases)}",
        f"  Distribución por clase     :",
    ]
    for c in clases:
        n = (y_true == c).sum()
        lineas.append(f"    {c:12s}: {n} docs")

    lineas += [
        "",
        "─" * 65,
        "NAIVE BAYES",
        "─" * 65,
        f"  Accuracy (real):   {acc_nb:.4f}  ({acc_nb:.1%})",
        f"  F1 macro (real):   {f1_nb:.4f}",
        "",
        "  Reporte por clase:",
        classification_report(y_true, y_nb, labels=clases,
                              target_names=clases, digits=4,
                              zero_division=0),
        "",
        "─" * 65,
        "SVM LINEAL",
        "─" * 65,
        f"  Accuracy (real):   {acc_svm:.4f}  ({acc_svm:.1%})",
        f"  F1 macro (real):   {f1_svm:.4f}",
        "",
        "  Reporte por clase:",
        classification_report(y_true, y_svm, labels=clases,
                              target_names=clases, digits=4,
                              zero_division=0),
        "",
        "─" * 65,
        "ERRORES DE CLASIFICACIÓN",
        "─" * 65,
    ]

    errores = df[~df["svm_ok"]][["archivo","clase_real","clase_svm","conf_svm"]]
    if errores.empty:
        lineas.append("  Ningún error en SVM.")
    else:
        for _, r in errores.iterrows():
            lineas.append(f"  {r['archivo'][:40]:<40} "
                          f"real={r['clase_real']:<12} "
                          f"pred={r['clase_svm']:<12} "
                          f"conf={r['conf_svm']:.1%}")

    lineas += ["", "=" * 65]
    texto = "\n".join(lineas)
    ruta.write_text(texto, encoding="utf-8")
    print(f"  ✓ {ruta.name}")
    print()
    print(texto)


# ─── Función principal ────────────────────────────────────────────────────────

def evaluar_real(docs: list[tuple[Path, str]]) -> pd.DataFrame:
    """Punto de entrada: recibe lista (pdf_path, clase_real), genera todo."""
    if not docs:
        print("[ERROR] No se proporcionaron documentos.")
        sys.exit(1)

    REPORTS_REAL.mkdir(parents=True, exist_ok=True)

    print("\n" + "═"*60)
    print("  EVALUACIÓN CON DOCUMENTOS REALES")
    print("═"*60)

    pipeline_nb, pipeline_svm = _cargar_modelos()

    # ── Evaluación ────────────────────────────────────────────────────────────
    df = evaluar(docs, pipeline_nb, pipeline_svm)

    if df.empty:
        print("[ERROR] No se pudo procesar ningún documento.")
        sys.exit(1)

    # ── Guardar CSV de resultados ─────────────────────────────────────────────
    csv_ruta = REPORTS_REAL / "resultados_reales.csv"
    df.to_csv(csv_ruta, index=False, encoding="utf-8")
    print(f"\n  ✓ Tabla de resultados → {csv_ruta}")

    # ── Figuras ───────────────────────────────────────────────────────────────
    print("\n  Generando figuras...")

    _figura_confusion(df["clase_real"], df["clase_nb"],
                      "Naive Bayes",
                      REPORTS_REAL / "confusion_nb_real.png")
    _figura_confusion(df["clase_real"], df["clase_svm"],
                      "SVM Lineal",
                      REPORTS_REAL / "confusion_svm_real.png")
    _figura_metricas(df, REPORTS_REAL / "comparacion_metricas_real.png")
    _figura_por_clase(df, REPORTS_REAL / "f1_por_clase_real.png")
    _figura_confianza(df, REPORTS_REAL / "confianza_real.png")

    # ── Reporte de texto ──────────────────────────────────────────────────────
    print()
    _reporte_texto(df, REPORTS_REAL / "reporte_evaluacion_real.txt")

    print("\n" + "═"*60)
    print(f"  ✓ Todos los archivos guardados en: {REPORTS_REAL}")
    print("═"*60)

    return df


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluación del clasificador con documentos PDF reales",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i",
        help="Carpeta con subcarpetas por clase (factura/, contrato/, informe/, correo/)",
        default=None,
    )
    parser.add_argument(
        "--pdf", nargs=2, metavar=("RUTA", "CLASE"),
        action="append", dest="pdfs",
        help="PDF individual con su clase real. Puede repetirse: --pdf factura.pdf factura",
    )
    parser.add_argument(
        "--lista", "-l",
        help="Archivo CSV con columnas 'ruta,clase' para evaluar en lote",
        default=None,
    )
    args = parser.parse_args()

    docs: list[tuple[Path, str]] = []

    # ── Opción A: carpeta organizada ──────────────────────────────────────────
    if args.input:
        carpeta = Path(args.input)
        if not carpeta.exists():
            print(f"[ERROR] Carpeta no encontrada: {carpeta}")
            sys.exit(1)
        print(f"\n  Buscando PDFs en {carpeta}...")
        docs += _recolectar_desde_carpeta(carpeta)

    # ── Opción B: PDFs individuales ────────────────────────────────────────────
    if args.pdfs:
        for ruta_str, clase in args.pdfs:
            ruta = Path(ruta_str)
            if not ruta.exists():
                print(f"  [WARN] No encontrado: {ruta}")
                continue
            if clase not in CLASES:
                print(f"  [WARN] Clase inválida '{clase}'. Válidas: {CLASES}")
                continue
            docs.append((ruta, clase))
            print(f"  + {ruta.name} → {clase}")

    # ── Opción C: CSV de lista ────────────────────────────────────────────────
    if args.lista:
        lista_path = Path(args.lista)
        if not lista_path.exists():
            print(f"[ERROR] Archivo no encontrado: {lista_path}")
            sys.exit(1)
        df_lista = pd.read_csv(lista_path)
        for _, row in df_lista.iterrows():
            ruta = Path(row["ruta"])
            if ruta.exists() and row["clase"] in CLASES:
                docs.append((ruta, row["clase"]))

    if not docs:
        parser.print_help()
        print("\n[ERROR] No se especificaron documentos. Ver uso arriba.")
        sys.exit(1)

    evaluar_real(docs)


if __name__ == "__main__":
    main()
