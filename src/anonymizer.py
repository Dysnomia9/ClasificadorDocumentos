"""
Módulo de anonimización de documentos reales.

Elimina información sensible antes de usar documentos reales como datos de entrenamiento:
  - RUTs chilenos (XX.XXX.XXX-X o XXXXXXXX-X)
  - Montos y valores ($XX.XXX.XXX)
  - Nombres propios detectados por patrones
  - Números de cédula / folio / cuenta
  - Correos electrónicos
  - Números de teléfono
  - Fechas específicas (reemplaza por [FECHA])
  - Direcciones (reemplaza por [DIRECCIÓN])

Uso:
    python -m src.anonymizer --input carpeta_docs/ --clase contrato
    python -m src.anonymizer --input mi_factura.pdf --clase factura

El módulo reemplaza los datos sensibles con tokens genéricos y
guarda el texto anonimizado como PDF en data/real/<clase>/
"""

import re
import sys
import argparse
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent.parent))

import fitz
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_JUSTIFY

from config import DATA_RAW_DIR, DATASET_CSV, CLASES


# ─── Patrones de anonimización ────────────────────────────────────────────────

PATRONES = [
    # RUT con puntos (12.345.678-9 o 12.345.678-K)
    (re.compile(r'\b\d{1,2}\.\d{3}\.\d{3}-[\dkKxX]\b'), '[RUT]'),
    # RUT sin puntos (12345678-9)
    (re.compile(r'\b\d{7,8}-[\dkKxX]\b'), '[RUT]'),
    # Correos electrónicos
    (re.compile(r'\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b', re.IGNORECASE), '[CORREO]'),
    # Teléfonos chilenos (+56 9 XXXX XXXX, 9XXXXXXXX, 2XXXXXXX)
    (re.compile(r'(?:\+56\s?)?(?:9\d{8}|2\d{7,8})\b'), '[TELÉFONO]'),
    # Montos en pesos ($X.XXX o $X.XXX.XXX)
    (re.compile(r'\$\s?\d{1,3}(?:\.\d{3})+'), '[MONTO]'),
    # Montos simples ($1000)
    (re.compile(r'\$\s?\d{3,}'), '[MONTO]'),
    # Números de cuenta bancaria (10-16 dígitos seguidos)
    (re.compile(r'\b\d{10,16}\b'), '[NÚMERO]'),
    # Folios y numeraciones largas
    (re.compile(r'\bN[°ºo]?\s?\d{4,}\b', re.IGNORECASE), 'N°[FOLIO]'),
    # Fechas (DD/MM/YYYY, DD-MM-YYYY, DD de mes de YYYY)
    (re.compile(
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|'
        r'\b\d{1,2}\s+de\s+(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|'
        r'septiembre|octubre|noviembre|diciembre)\s+de\s+\d{4}\b',
        re.IGNORECASE
    ), '[FECHA]'),
    # Direcciones comunes (Av., Calle, Jr., etc. + número)
    (re.compile(
        r'(?:Av(?:enida)?|Calle|Jr\.?|Pje\.?|Pasaje|Camino|Ruta)\s+[\w\s]+?\s+\d+',
        re.IGNORECASE
    ), '[DIRECCIÓN]'),
    # Nombres propios en contextos típicos (después de don/doña/señor/señora)
    (re.compile(
        r'(?:don|doña|señor(?:a)?|sr\.?|sra\.?)\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){1,3}',
        re.IGNORECASE
    ), '[NOMBRE]'),
    # Nombres propios: mínimo 3 palabras en mayúscula inicial (excluye encabezados
    # en MAYÚSCULAS completas como "CONTRATO DE TRABAJO" que tienen >1 letra minúscula)
    (re.compile(
        r'\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]{1,}(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]{1,}){2,3}\b'
    ), '[NOMBRE_PROPIO]'),
]

# Limpieza adicional de espacios múltiples
RE_ESPACIOS = re.compile(r'  +')


def anonimizar_texto(texto: str) -> tuple[str, dict]:
    """
    Aplica todos los patrones de anonimización a un texto.

    Retorna (texto_anonimizado, estadísticas_de_reemplazos).
    """
    stats = {}
    resultado = texto

    for patron, reemplazo in PATRONES:
        matches = patron.findall(resultado)
        if matches:
            stats[reemplazo] = len(matches)
        resultado = patron.sub(reemplazo, resultado)

    resultado = RE_ESPACIOS.sub(' ', resultado)
    return resultado, stats


def anonimizar_pdf(ruta_entrada: Path, ruta_salida: Path) -> dict:
    """
    Lee un PDF, anonimiza su texto y guarda el resultado como nuevo PDF.

    Retorna diccionario con estadísticas de lo que se reemplazó.
    """
    # Extraer texto del original
    doc = fitz.open(str(ruta_entrada))
    texto_original = "\n".join(p.get_text("text") for p in doc)
    doc.close()

    if not texto_original.strip():
        return {"error": "PDF sin texto extraíble (posiblemente escaneado)"}

    # Anonimizar
    texto_anon, stats = anonimizar_texto(texto_original)

    # Guardar como nuevo PDF
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    _texto_a_pdf(texto_anon, ruta_salida)

    return stats


def _texto_a_pdf(texto: str, ruta: Path) -> None:
    """Guarda texto plano como PDF con formato básico."""
    doc = SimpleDocTemplate(str(ruta), pagesize=A4,
                            leftMargin=2.5*cm, rightMargin=2.5*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    estilo = ParagraphStyle("body", fontSize=9, leading=14,
                            alignment=TA_JUSTIFY, spaceAfter=6)
    story = []
    for linea in texto.split('\n'):
        linea = linea.strip()
        if linea:
            # Escapar caracteres especiales de ReportLab
            linea = linea.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            try:
                story.append(Paragraph(linea, estilo))
            except Exception:
                pass
        else:
            story.append(Spacer(1, 0.2*cm))
    if not story:
        story.append(Paragraph("(documento vacío tras anonimización)", estilo))
    doc.build(story)


def procesar_carpeta(
    carpeta_entrada: Path,
    clase: str,
    verbose: bool = True
) -> list[dict]:
    """
    Anonimiza todos los PDFs de una carpeta y los agrega al corpus.

    Los PDFs anonimizados se guardan en data/real/<clase>/
    y se registran en dataset.csv con la columna 'fuente'='real'.

    Parámetros
    ----------
    carpeta_entrada : carpeta con los PDFs originales (no se modifican)
    clase           : categoría documental ('factura','contrato','informe','correo')
    verbose         : imprimir progreso

    Retorna
    -------
    Lista de registros {path, clase, fuente} agregados al dataset.
    """
    if clase not in CLASES:
        raise ValueError(f"Clase '{clase}' inválida. Debe ser una de: {CLASES}")

    pdfs = sorted(carpeta_entrada.glob("*.pdf"))
    if not pdfs:
        print(f"[WARN] No se encontraron PDFs en {carpeta_entrada}")
        return []

    carpeta_salida = DATA_RAW_DIR / clase
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    # Calcular índice de inicio para no sobreescribir existentes
    existentes = list(carpeta_salida.glob("real_*.pdf"))
    idx_inicio = len(existentes)

    registros = []
    if verbose:
        print(f"\n  Anonimizando {len(pdfs)} PDFs de clase '{clase}'...")
        print(f"  Destino: {carpeta_salida}")

    for i, pdf_orig in enumerate(pdfs):
        nombre_salida = carpeta_salida / f"real_{idx_inicio + i:04d}.pdf"
        try:
            stats = anonimizar_pdf(pdf_orig, nombre_salida)
            if "error" in stats:
                if verbose:
                    print(f"  [SKIP] {pdf_orig.name}: {stats['error']}")
                continue

            registros.append({
                "path":   str(nombre_salida),
                "clase":  clase,
                "fuente": "real",
                "tipo":   "real",
            })
            if verbose:
                reemplazos = sum(v for k,v in stats.items() if k != 'error')
                print(f"  ✓ {pdf_orig.name} → {nombre_salida.name} "
                      f"({reemplazos} reemplazos: {stats})")
        except Exception as e:
            if verbose:
                print(f"  [ERROR] {pdf_orig.name}: {e}")

    # Actualizar dataset.csv
    if registros and DATASET_CSV.exists():
        df_existente = pd.read_csv(DATASET_CSV)
        df_nuevos = pd.DataFrame(registros)
        # Agregar columnas faltantes
        for col in df_existente.columns:
            if col not in df_nuevos.columns:
                df_nuevos[col] = None
        df_total = pd.concat([df_existente, df_nuevos], ignore_index=True)
        df_total.to_csv(DATASET_CSV, index=False)
        if verbose:
            print(f"\n  ✓ {len(registros)} documentos reales agregados al dataset")
            print(f"  Dataset total: {len(df_total)} documentos")
    elif registros:
        pd.DataFrame(registros).to_csv(DATASET_CSV, index=False)

    return registros


def demo_anonimizacion(texto: str) -> None:
    """Muestra en consola cómo quedaría un texto tras anonimizar."""
    anon, stats = anonimizar_texto(texto)
    print("ORIGINAL:")
    print(texto[:500])
    print("\nANONIMIZADO:")
    print(anon[:500])
    print(f"\nReemplazos: {stats}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Anonimizador de documentos reales para entrenamiento del clasificador"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Carpeta con PDFs a anonimizar (o ruta a un PDF individual)"
    )
    parser.add_argument(
        "--clase", "-c", required=True, choices=CLASES,
        help="Clase documental de los PDFs"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Solo mostrar un ejemplo de anonimización sin guardar"
    )
    args = parser.parse_args()

    ruta = Path(args.input)

    if args.demo:
        texto_demo = (
            "CONTRATO DE TRABAJO\n"
            "En Santiago de Chile, a 15 de marzo de 2024, entre Empresa XYZ Ltda., "
            "RUT 76.123.456-7, representada por don Juan Carlos Pérez González, "
            "domiciliada en Av. Providencia 1234, teléfono +56 9 87654321, "
            "correo: jperez@empresa.cl; y el Trabajador don María Fernanda López Soto, "
            "RUT 15.678.901-K, se conviene el siguiente contrato con remuneración "
            "de $850.000 mensuales."
        )
        demo_anonimizacion(texto_demo)
    elif ruta.is_dir():
        procesar_carpeta(ruta, args.clase, verbose=True)
    elif ruta.is_file() and ruta.suffix.lower() == ".pdf":
        salida = DATA_RAW_DIR / args.clase / f"real_{ruta.stem}_anon.pdf"
        stats = anonimizar_pdf(ruta, salida)
        print(f"✓ Anonimizado: {salida}")
        print(f"  Reemplazos: {stats}")
    else:
        print(f"[ERROR] {ruta} no es una carpeta ni un PDF válido")
