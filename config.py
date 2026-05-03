"""
Configuración central del proyecto.
Modifica este archivo para ajustar rutas, parámetros y clases.
"""

from pathlib import Path

# ─── Rutas del proyecto ─────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
DATA_RAW_DIR  = BASE_DIR / "data" / "raw"
DATA_PROC_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR    = BASE_DIR / "models"
REPORTS_DIR   = BASE_DIR / "reports"

# ─── Clases de documentos ────────────────────────────────────────────────────
CLASES = ["factura", "contrato", "informe", "cotizacion", "orden_compra"]

CLASE_LABEL = {
    "factura":      0,
    "contrato":     1,
    "informe":      2,
    "cotizacion":   3,
    "orden_compra": 4,
}

LABEL_CLASE = {v: k for k, v in CLASE_LABEL.items()}

# ─── Generación de datos ─────────────────────────────────────────────────────
DOCS_POR_CLASE = 145

# ─── Modelos ─────────────────────────────────────────────────────────────────
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE  = (1, 1)
TEST_SIZE          = 0.20
RANDOM_STATE       = 42

# ─── Archivos de salida ───────────────────────────────────────────────────────
DATASET_CSV    = DATA_PROC_DIR / "dataset.csv"
VECTORIZER_PKL = MODELS_DIR    / "vectorizer.pkl"
NB_MODEL_PKL   = MODELS_DIR    / "naive_bayes.pkl"
SVM_MODEL_PKL  = MODELS_DIR    / "svm.pkl"
REPORT_TXT     = REPORTS_DIR   / "reporte_comparacion.txt"

# ─── Tipos de archivo soportados ─────────────────────────────────────────────
EXTENSIONES_IMAGEN = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}
EXTENSIONES_PDF    = {".pdf"}

DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROC_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
