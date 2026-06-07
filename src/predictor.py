"""
Módulo de predicción para documentos nuevos.

Uso:
  from src.predictor import clasificar_documento

  resultado = clasificar_documento("mi_factura.pdf")
  # {'clase': 'factura', 'confianza': 0.97, 'modelo': 'SVM', ...}
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
from config import NB_MODEL_PKL, SVM_MODEL_PKL


def _cargar_modelos():
    """Carga los pipelines entrenados desde disco."""
    if not NB_MODEL_PKL.exists() or not SVM_MODEL_PKL.exists():
        raise FileNotFoundError(
            "No se encontraron los modelos entrenados.\n"
            "Ejecuta primero:  python main.py train"
        )
    return joblib.load(NB_MODEL_PKL), joblib.load(SVM_MODEL_PKL)


def clasificar_documento(pdf_path: str | Path,
                          modelo: str = "svm",
                          verbose: bool = True) -> dict:
    """
    Clasifica un documento PDF y retorna la clase predicha.

    Parámetros
    ----------
    pdf_path : ruta al PDF a clasificar
    modelo   : 'svm' | 'nb' | 'ambos'
    verbose  : imprime resultado en consola

    Retorna
    -------
    dict con:
        clase       : clase predicha (str)
        confianza   : probabilidad/decisión (float, 0-1)
        detalle     : scores por clase (dict)
        modelo_usado: nombre del modelo aplicado
    """
    from src.extractor import extraer_texto_pdf
    from src.preprocessor import limpiar_texto

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {pdf_path}")

    pipeline_nb, pipeline_svm = _cargar_modelos()
    pipeline_activo = pipeline_svm if modelo.lower() in ("svm", "ambos") else pipeline_nb
    nombre_modelo   = "SVM" if modelo.lower() in ("svm", "ambos") else "Naive Bayes"

    # Extraer y preprocesar texto
    texto_raw    = extraer_texto_pdf(pdf_path)
    texto_limpio = limpiar_texto(texto_raw)

    if not texto_limpio.strip():
        return {
            "clase":        "desconocido",
            "confianza":    0.0,
            "detalle":      {},
            "modelo_usado": nombre_modelo,
            "error":        "No se pudo extraer texto del documento.",
        }

    # Predicción con SVM (no tiene predict_proba)
    clase_pred = pipeline_activo.predict([texto_limpio])[0]

    # Obtener scores de decisión del SVM
    detalle = {}
    confianza = 1.0

    if hasattr(pipeline_activo.named_steps["clf"], "decision_function"):
        scores = pipeline_activo.named_steps["clf"].decision_function(
            pipeline_activo.named_steps["tfidf"].transform([texto_limpio])
        )[0]
        clases = pipeline_activo.classes_
        # Normalizar con softmax para interpretabilidad
        e_scores  = np.exp(scores - scores.max())
        probs      = e_scores / e_scores.sum()
        detalle    = {c: round(float(p), 4) for c, p in zip(clases, probs)}
        confianza  = float(probs.max())
    elif hasattr(pipeline_activo.named_steps["clf"], "predict_proba"):
        probs      = pipeline_activo.predict_proba([texto_limpio])[0]
        clases     = pipeline_activo.classes_
        detalle    = {c: round(float(p), 4) for c, p in zip(clases, probs)}
        confianza  = float(probs.max())

    resultado = {
        "clase":        clase_pred,
        "confianza":    round(confianza, 4),
        "detalle":      detalle,
        "modelo_usado": nombre_modelo,
        "archivo":      str(pdf_path),
    }

    if verbose:
        print(f"\n{'─'*50}")
        print(f"  Archivo:  {pdf_path.name}")
        print(f"  Modelo:   {nombre_modelo}")
        print(f"  Clase:    {clase_pred.upper()}  (confianza: {confianza:.2%})")
        if detalle:
            print(f"  Scores por clase:")
            for c, s in sorted(detalle.items(), key=lambda x: -x[1]):
                barra = "█" * int(s * 20)
                print(f"    {c:12s} {s:.4f}  {barra}")
        print(f"{'─'*50}")

    # Comparar ambos modelos si se solicita
    if modelo.lower() == "ambos":
        clase_nb  = pipeline_nb.predict([texto_limpio])[0]
        clase_svm = pipeline_svm.predict([texto_limpio])[0]
        resultado["nb_pred"]  = clase_nb
        resultado["svm_pred"] = clase_svm
        if verbose:
            print(f"\n  Comparación:")
            print(f"    Naive Bayes: {clase_nb}")
            print(f"    SVM:         {clase_svm}")
            coinciden = "✓ Coinciden" if clase_nb == clase_svm else "⚠ Difieren"
            print(f"    {coinciden}")

    return resultado


def clasificar_lote(lista_pdfs: list, modelo: str = "svm") -> list:
    """
    Clasifica una lista de PDFs y retorna lista de resultados.

    Útil para evaluar sobre el dataset de test o clasificar carpetas completas.
    """
    resultados = []
    for i, pdf_path in enumerate(lista_pdfs):
        r = clasificar_documento(pdf_path, modelo=modelo, verbose=False)
        resultados.append(r)
        if (i + 1) % 20 == 0:
            print(f"  Clasificados {i+1}/{len(lista_pdfs)}")
    return resultados


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        resultado = clasificar_documento(sys.argv[1], modelo="ambos")
    else:
        print("Uso: python predictor.py <ruta_documento.pdf>")
