"""
Módulo de extracción de texto desde PDFs e imágenes.

Soporta dos tipos de entrada:
  A) Archivos PDF  → extracción directa con PyMuPDF; fallback OCR si es escaneado
  B) Imágenes      → OCR con Tesseract tras preprocesamiento (jpg, png, tiff, bmp, webp)

Función de entrada unificada:
  extraer_texto(ruta)   → detecta el tipo y llama al método correcto
  extraer_texto_pdf     → para PDFs
  extraer_texto_imagen  → para imágenes

Pipeline de preprocesamiento de imágenes:
  1. Escala de grises
  2. Reescalado (mínimo 1800 px de ancho para OCR de calidad)
  3. Realce de contraste
  4. Nitidez
  5. Tesseract con idioma español (spa)
"""

import sys
import io
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

import fitz          # PyMuPDF
import pandas as pd
from config import DATASET_CSV, DATA_PROC_DIR, EXTENSIONES_IMAGEN, EXTENSIONES_PDF

# ─── Disponibilidad de dependencias opcionales ────────────────────────────────

try:
    import pytesseract
    from PIL import Image, ImageEnhance, ImageFilter
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# ─── Configuración de Tesseract ────────────────────────────────────────────────
# Se detecta en tiempo de importación qué idiomas están disponibles.
# Prioridad: español (spa) → inglés (eng) como fallback.
# --oem 3  : motor LSTM + legado (mejor precisión general)
# --psm 6  : bloque uniforme de texto (adecuado para documentos)

def _detectar_idioma_tesseract() -> str:
    """Retorna 'spa' si está disponible, 'eng' como fallback."""
    if not TESSERACT_AVAILABLE:
        return "spa"   # no importa, no se usará
    try:
        idiomas = pytesseract.get_languages()
        if "spa" in idiomas:
            return "spa"
        print("[WARN] Paquete de español (spa) no instalado en Tesseract.")
        print("       Usando inglés (eng). Para mejor precisión en documentos chilenos:")
        print("       Linux: sudo apt install tesseract-ocr-spa")
        print("       macOS: brew install tesseract-lang")
        return "eng" if "eng" in idiomas else idiomas[0]
    except Exception:
        return "spa"

TESSERACT_LANG   = _detectar_idioma_tesseract() if TESSERACT_AVAILABLE else "spa"
TESSERACT_CONFIG = f"--oem 3 --psm 6 -l {TESSERACT_LANG}"


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESAMIENTO DE IMAGEN
# ══════════════════════════════════════════════════════════════════════════════

def _preprocesar_imagen(img: "Image.Image") -> "Image.Image":
    """
    Pipeline de preprocesamiento para mejorar la calidad del OCR.

    Pasos:
      1. Conversión a escala de grises
      2. Reescalado al mínimo 1 800 px de ancho (mejora OCR en fotos de celular)
      3. Realce de contraste (factor 1.8)
      4. Filtro de nitidez
    """
    from PIL import ImageEnhance, ImageFilter

    # 1. Escala de grises
    img = img.convert("L")

    # 2. Reescalar si la imagen es pequeña
    ancho, alto = img.size
    min_ancho   = 1_800
    if ancho < min_ancho:
        factor = min_ancho / ancho
        nuevo_ancho = min_ancho
        nuevo_alto  = int(alto * factor)
        img = img.resize((nuevo_ancho, nuevo_alto), Image.LANCZOS)

    # 3. Realce de contraste
    img = ImageEnhance.Contrast(img).enhance(1.8)

    # 4. Nitidez
    img = img.filter(ImageFilter.SHARPEN)

    return img


def _rotar_si_necesario(img: "Image.Image") -> "Image.Image":
    """
    Aplica la rotación EXIF correcta si la imagen viene de un celular.
    Evita texto rotado que confunde a Tesseract.
    """
    try:
        from PIL import ExifTags
        exif = img._getexif()
        if exif is None:
            return img
        for tag, value in exif.items():
            if ExifTags.TAGS.get(tag) == "Orientation":
                if   value == 3: img = img.rotate(180, expand=True)
                elif value == 6: img = img.rotate(270, expand=True)
                elif value == 8: img = img.rotate(90,  expand=True)
                break
    except Exception:
        pass
    return img


# ══════════════════════════════════════════════════════════════════════════════
# EXTRACCIÓN DESDE IMAGEN
# ══════════════════════════════════════════════════════════════════════════════

def extraer_texto_imagen(ruta: str | Path) -> str:
    """
    Extrae texto de una imagen (JPG, PNG, TIFF, BMP, WEBP) usando Tesseract OCR.

    Parámetros
    ----------
    ruta : ruta al archivo de imagen

    Retorna
    -------
    str con el texto reconocido (vacío si Tesseract no está disponible).
    """
    if not TESSERACT_AVAILABLE:
        print("[WARN] pytesseract / Pillow no instalados. "
              "Ejecuta: pip install pytesseract Pillow")
        print("        Instala también Tesseract: "
              "https://github.com/UB-Mannheim/tesseract/wiki")
        return ""

    from PIL import Image

    try:
        img = Image.open(str(ruta))
        img = _rotar_si_necesario(img)
        img = _preprocesar_imagen(img)
        texto = pytesseract.image_to_string(img, config=TESSERACT_CONFIG)
        return texto.strip()
    except Exception as e:
        print(f"[WARN] OCR falló para {ruta}: {e}")
        return ""


def extraer_texto_imagen_bytes(data: bytes, nombre: str = "imagen") -> str:
    """
    Versión que acepta bytes directamente (útil para Streamlit file_uploader).

    Parámetros
    ----------
    data   : contenido del archivo como bytes
    nombre : nombre del archivo (solo para mensajes de error)
    """
    if not TESSERACT_AVAILABLE:
        print("[WARN] pytesseract / Pillow no instalados.")
        return ""

    from PIL import Image

    try:
        img = Image.open(io.BytesIO(data))
        img = _rotar_si_necesario(img)
        img = _preprocesar_imagen(img)
        texto = pytesseract.image_to_string(img, config=TESSERACT_CONFIG)
        return texto.strip()
    except Exception as e:
        print(f"[WARN] OCR falló para {nombre}: {e}")
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# EXTRACCIÓN DESDE PDF
# ══════════════════════════════════════════════════════════════════════════════

def extraer_texto_pdf(pdf_path: str | Path, min_chars: int = 50) -> str:
    """
    Extrae texto de un PDF usando PyMuPDF.
    Si el texto digital es insuficiente (PDF escaneado), aplica OCR página a página.

    Parámetros
    ----------
    pdf_path  : ruta al PDF
    min_chars : umbral mínimo de caracteres para considerar extracción exitosa

    Retorna
    -------
    str con el texto del documento.
    """
    texto = ""
    try:
        doc = fitz.open(str(pdf_path))
        for pagina in doc:
            texto += pagina.get_text("text")
        doc.close()
    except Exception as e:
        print(f"[WARN] Error leyendo {pdf_path}: {e}")
        return ""

    # Texto digital suficiente → retornar directamente
    if len(texto.strip()) >= min_chars:
        return texto.strip()

    # PDF escaneado → OCR página a página
    if TESSERACT_AVAILABLE:
        try:
            from PIL import Image

            doc      = fitz.open(str(pdf_path))
            ocr_text = ""
            for pagina in doc:
                mat = fitz.Matrix(2.0, 2.0)           # zoom 2x → mejor OCR
                pix = pagina.get_pixmap(matrix=mat)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                img = _preprocesar_imagen(img)
                ocr_text += pytesseract.image_to_string(img, config=TESSERACT_CONFIG)
            doc.close()
            return ocr_text.strip()
        except Exception as e:
            print(f"[WARN] OCR falló para {pdf_path}: {e}")

    return texto.strip()


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIÓN UNIFICADA
# ══════════════════════════════════════════════════════════════════════════════

def extraer_texto(ruta: str | Path) -> str:
    """
    Función unificada: detecta si el archivo es PDF o imagen y aplica
    el método de extracción adecuado.

    Extensiones soportadas:
      PDF   : .pdf
      Imagen: .jpg .jpeg .png .tiff .tif .bmp .webp
    """
    ruta = Path(ruta)
    ext  = ruta.suffix.lower()

    if ext in EXTENSIONES_IMAGEN:
        return extraer_texto_imagen(ruta)
    elif ext in EXTENSIONES_PDF:
        return extraer_texto_pdf(ruta)
    else:
        print(f"[WARN] Extensión no soportada: {ext} — intentando como PDF.")
        return extraer_texto_pdf(ruta)


# ══════════════════════════════════════════════════════════════════════════════
# CORPUS
# ══════════════════════════════════════════════════════════════════════════════

def extraer_corpus(df: pd.DataFrame | None = None,
                   verbose: bool = True) -> pd.DataFrame:
    """
    Itera sobre el DataFrame y extrae texto de cada archivo (PDF o imagen).
    Agrega columna 'texto' y guarda en dataset.csv.
    """
    if df is None:
        df = pd.read_csv(DATASET_CSV)

    if "texto" in df.columns:
        ya_extraidos = df["texto"].notna().sum()
        if ya_extraidos == len(df):
            if verbose:
                print("✓ Texto ya extraído previamente.")
            return df

    textos = []
    for i, row in df.iterrows():
        texto = extraer_texto(row["path"])
        textos.append(texto)
        if verbose and (i + 1) % 50 == 0:
            print(f"  → Procesados {i+1}/{len(df)} documentos...")

    df = df.copy()
    df["texto"] = textos

    n_vacios = (df["texto"].str.strip() == "").sum()
    if n_vacios > 0 and verbose:
        print(f"[WARN] {n_vacios} documentos sin texto extraído → se eliminarán.")
    df = df[df["texto"].str.strip() != ""].reset_index(drop=True)

    df.to_csv(DATASET_CSV, index=False)
    if verbose:
        print(f"✓ Extracción completa: {len(df)} documentos → {DATASET_CSV}")

    return df


def verificar_ocr() -> dict:
    """
    Verifica qué dependencias de OCR están disponibles.
    Útil para diagnóstico antes de procesar imágenes.
    """
    estado = {
        "pymupdf":   True,
        "pillow":    PIL_AVAILABLE,
        "tesseract": TESSERACT_AVAILABLE,
    }
    if TESSERACT_AVAILABLE:
        try:
            estado["tesseract_version"] = str(pytesseract.get_tesseract_version())
            idiomas = pytesseract.get_languages()
            estado["spa_disponible"] = "spa" in idiomas
            estado["idioma_activo"]  = TESSERACT_LANG
            estado["config_activa"]  = TESSERACT_CONFIG
        except Exception as e:
            estado["tesseract_error"] = str(e)
    return estado


if __name__ == "__main__":
    print("=== Verificación de dependencias OCR ===")
    est = verificar_ocr()
    for k, v in est.items():
        icono = "✓" if v is True else ("✗" if v is False else "→")
        print(f"  {icono} {k}: {v}")

    if not est.get("tesseract"):
        print("\n  Para habilitar OCR en imágenes y PDFs escaneados:")
        print("  1. pip install pytesseract Pillow")
        print("  2. Instalar Tesseract: https://github.com/UB-Mannheim/tesseract/wiki")
        print("  3. En Windows, agregar al PATH: C:\\Program Files\\Tesseract-OCR")
        print("  4. Descargar paquete de idioma español (spa.traineddata)")
