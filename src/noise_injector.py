"""
Módulo de inyección de ruido — versión corregida v2.

CORRECCIÓN: Solo se inyecta ruido en documentos de entrenamiento ('split' == 'train').
Los documentos de test conservan su integridad (frases holdout sin modificar),
preservando la separación estricta que garantiza métricas honestas.

El ruido genera documentos genuinamente ambiguos mezclando:
  - Párrafos de dos clases distintas (50%)
  - Documentos cortos con poca señal (25%)
  - Documentos donde el tono es de una clase y el contenido de otra (25%)
"""
import random
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_JUSTIFY

from config import DATASET_CSV, CLASES, RANDOM_STATE

rng = random.Random(RANDOM_STATE + 13)

# ─── Frases de ruido — importadas del pool de entrenamiento ──────────────────
# Se importan en tiempo de ejecución para evitar importación circular.

def _get_frases_train():
    """Carga el pool FRASES_TRAIN desde realistic_corpus si está disponible."""
    try:
        from src.realistic_corpus import FRASES_TRAIN, NEUTRAS_TRAIN
        return FRASES_TRAIN, NEUTRAS_TRAIN
    except ImportError:
        pass
    try:
        from realistic_corpus import FRASES_TRAIN, NEUTRAS_TRAIN
        return FRASES_TRAIN, NEUTRAS_TRAIN
    except ImportError:
        return None, None


# Pool de respaldo si no se puede importar realistic_corpus
FRASES_RESPALDO = {
    "contrato": [
        "El Empleador contrata al Trabajador para desempeñar el cargo conforme a las condiciones laborales.",
        "La jornada ordinaria de trabajo será de cuarenta y cinco horas semanales distribuidas de lunes a viernes.",
        "La remuneración mensual bruta del Trabajador se pagará dentro de los primeros cinco días hábiles.",
        "El Trabajador se obliga a guardar reserva y confidencialidad sobre los negocios de la Empresa.",
        "El Arrendador da en arrendamiento al Arrendatario el inmueble ubicado en la dirección indicada.",
        "El presente instrumento se firma en dos ejemplares del mismo tenor y fecha por las partes.",
    ],
    "factura": [
        "La presente factura electrónica da cuenta de los servicios prestados durante el período.",
        "El monto neto más el impuesto al valor agregado del diecinueve por ciento resulta en el total.",
        "El emisor certifica que los montos facturados corresponden efectivamente a los servicios entregados.",
        "El receptor deberá efectuar el pago dentro del plazo de vencimiento según condiciones acordadas.",
        "El folio identificador asignado por el SII individualiza este documento tributario en el sistema.",
    ],
    "informe": [
        "Los indicadores clave de desempeño reflejan un nivel de cumplimiento dentro de los parámetros.",
        "El análisis comparativo muestra una tendencia positiva respecto al período anterior evaluado.",
        "Las recomendaciones presentadas se basan en el análisis cuantitativo de los indicadores.",
        "El equipo técnico elaboró el presente reporte en base a la información del período evaluado.",
        "Se identificaron áreas de mejora que requieren atención prioritaria por la gerencia operacional.",
    ],
    "correo": [
        "Me dirijo a usted para solicitar formalmente información sobre los servicios disponibles.",
        "En respuesta a su consulta adjunto la documentación solicitada para su revisión y tramitación.",
        "Confirmo la recepción del documento enviado y agradezco su gestión oportuna al respecto.",
        "Le comunico formalmente la decisión adoptada por la organización respecto al asunto indicado.",
        "Quedamos en espera de su confirmación a la brevedad para proceder con las gestiones necesarias.",
    ],
}

NEUTRAS_RESPALDO = [
    "En conformidad con lo establecido, adjuntamos los antecedentes correspondientes para su revisión.",
    "Le rogamos tomar nota de lo indicado y proceder según los procedimientos internos vigentes.",
    "El presente documento ha sido preparado conforme a los estándares establecidos por la organización.",
    "Para mayor información no dude en contactarse con el área responsable a través de los canales.",
]


def _write_pdf(path: Path, texto: str) -> None:
    doc = SimpleDocTemplate(str(path), pagesize=A4,
                            leftMargin=2.5*cm, rightMargin=2.5*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    estilo = ParagraphStyle("b", fontSize=9, leading=14,
                            alignment=TA_JUSTIFY, spaceAfter=6)
    story = []
    for linea in texto.split("\n"):
        l = linea.strip()
        if l:
            l = l.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            try:
                story.append(Paragraph(l, estilo))
            except Exception:
                pass
        else:
            story.append(Spacer(1, 0.2*cm))
    if not story:
        story.append(Paragraph("Documento.", estilo))
    doc.build(story)


def _doc_ambiguo(clase: str, frases: dict, neutras: list) -> str:
    """Documento con 35-55% de frases de otra clase."""
    otras = [c for c in CLASES if c != clase]
    clase_contaminante = rng.choice(otras)

    frac_cont = rng.uniform(0.35, 0.55)
    n_total   = rng.randint(10, 16)
    n_cont    = max(2, int(n_total * frac_cont))
    n_propia  = n_total - n_cont
    n_neutra  = rng.randint(1, 3)

    partes = (
        rng.choices(frases[clase],              k=max(1, n_propia)) +
        rng.choices(frases[clase_contaminante], k=n_cont) +
        rng.choices(neutras,                    k=n_neutra)
    )
    rng.shuffle(partes)
    return "\n".join(partes)


def _doc_corto(clase: str, frases: dict) -> str:
    """Documento corto (3-5 frases) — poca señal discriminativa."""
    n = rng.randint(3, 5)
    return "\n".join(rng.choices(frases[clase], k=n))


def _doc_estructura_invertida(clase: str, frases: dict, neutras: list) -> str:
    """Estructura de otra clase mezclada con contenido de la clase real."""
    otras = [c for c in CLASES if c != clase]
    clase_tono = rng.choice(otras)

    frases_contenido = rng.choices(frases[clase],      k=rng.randint(7, 10))
    frases_tono      = rng.choices(frases[clase_tono], k=rng.randint(3, 5))
    n_neutras        = rng.choices(neutras,             k=2)

    todas = frases_tono[:2] + frases_contenido + n_neutras + frases_tono[2:]
    return "\n".join(todas)


_TIPOS_RUIDO = [
    ("ambiguo",          0.50),
    ("corto",            0.25),
    ("estructura_inv",   0.25),
]


def inyectar_ruido_dataset(
    fraccion_ruidosos: float = 0.40,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Reemplaza una fracción de los documentos de ENTRENAMIENTO con versiones
    genuinamente ambiguas. Los documentos de TEST NO se modifican.

    Parámetros
    ----------
    fraccion_ruidosos : fracción del conjunto de TRAIN a contaminar
    verbose           : imprime progreso
    """
    df = pd.read_csv(DATASET_CSV)

    # Cargar frases desde realistic_corpus o usar el respaldo
    frases_train, neutras_train = _get_frases_train()
    if frases_train is None:
        if verbose:
            print("  ⚠  realistic_corpus no disponible — usando frases de respaldo")
        frases_train = FRASES_RESPALDO
        neutras_train = NEUTRAS_RESPALDO

    # ── CRÍTICO: solo contaminar documentos de TRAIN ─────────────────────────
    if "split" in df.columns:
        indices_train = df.index[df["split"] == "train"].tolist()
        if verbose:
            n_test = (df["split"] == "test").sum()
            print(f"  → Documentos de TEST protegidos: {n_test} (no se modifican)")
    else:
        indices_train = df.index.tolist()
        if verbose:
            print("  ⚠  Columna 'split' no encontrada — aplicando ruido a todos los docs")

    n_ruidosos = int(len(indices_train) * fraccion_ruidosos)
    indices_ruido = rng.sample(indices_train, k=n_ruidosos)

    if verbose:
        print(f"  → Inyectando ruido en {n_ruidosos}/{len(indices_train)} "
              f"docs de train ({fraccion_ruidosos:.0%})...")

    df["ruidoso"] = False
    tipos_count = {"ambiguo": 0, "corto": 0, "estructura_inv": 0}

    for i_proc, idx in enumerate(indices_ruido):
        row   = df.iloc[idx]
        clase = row["clase"]
        path  = Path(row["path"])

        # Elegir tipo de ruido
        r = rng.random()
        acum = 0.0
        tipo_elegido = "ambiguo"
        fn_map = {
            "ambiguo":        lambda c: _doc_ambiguo(c, frases_train, neutras_train),
            "corto":          lambda c: _doc_corto(c, frases_train),
            "estructura_inv": lambda c: _doc_estructura_invertida(c, frases_train, neutras_train),
        }
        for tipo_key, prob in _TIPOS_RUIDO:
            acum += prob
            if r < acum:
                tipo_elegido = tipo_key
                break

        try:
            texto = fn_map[tipo_elegido](clase)
            _write_pdf(path, texto)
            df.at[idx, "ruidoso"] = True
            tipos_count[tipo_elegido] += 1
        except Exception as e:
            if verbose:
                print(f"    [WARN] idx {idx}: {e}")

        if verbose and (i_proc + 1) % 60 == 0:
            print(f"    Procesados {i_proc+1}/{n_ruidosos}...")

    # Limpiar texto extraído de docs modificados para forzar re-extracción
    df.loc[df["ruidoso"], "texto"] = None
    if "texto_limpio" in df.columns:
        df.loc[df["ruidoso"], "texto_limpio"] = None

    df.to_csv(DATASET_CSV, index=False)

    if verbose:
        print(f"  ✓ Ruido inyectado en documentos de TRAIN:")
        print(f"     Ambiguos (35-55% de otra clase): {tipos_count['ambiguo']}")
        print(f"     Cortos (poca señal):              {tipos_count['corto']}")
        print(f"     Estructura invertida:             {tipos_count['estructura_inv']}")
        print(f"  Dataset guardado → {DATASET_CSV}")

    return df


if __name__ == "__main__":
    print("Inyectando ruido semántico (solo en documentos de train)...")
    df = inyectar_ruido_dataset(fraccion_ruidosos=0.40, verbose=True)
    print(f"\nRuidosos (train): {df['ruidoso'].sum()} | Limpios: {(~df['ruidoso']).sum()}")
    if "split" in df.columns:
        print(f"Test intactos:    {(df['split'] == 'test').sum()}")
