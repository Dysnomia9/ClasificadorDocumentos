"""
Clasificador Automático de Documentos Empresariales
====================================================
Interfaz web Streamlit del clasificador de documentos

Soporta:
  - Archivos PDF (texto digital + fallback OCR para escaneados)
  - Imágenes (JPG, PNG, TIFF, BMP, WEBP) → OCR con Tesseract

Ejecutar:
    streamlit run app.py
"""

import sys
import time
import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

# ─── Página ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Clasificador de Documentos",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Rutas ────────────────────────────────────────────────────────────────────
RESULTADOS_DIR = Path(__file__).parent / "resultados"

# ─── Paleta y descripciones (5 clases) ───────────────────────────────────────
COLORES = {
    "factura":      "#2980b9",
    "contrato":     "#e74c3c",
    "informe":      "#27ae60",
    "cotizacion":   "#f39c12",
    "orden_compra": "#16a085",
}
DESCRIPCIONES = {
    "factura":      "Factura o boleta electrónica con datos tributarios, montos e IVA.",
    "contrato":     "Contrato de servicios, arrendamiento, trabajo u otro acuerdo entre partes.",
    "informe":      "Informe de gestión, reporte técnico o documento de resultados.",
    "cotizacion":   "Cotización o presupuesto comercial con oferta de productos o servicios.",
    "orden_compra": "Orden de compra que autoriza la adquisición de bienes a un proveedor.",
}

# Extensiones de imagen aceptadas
EXTENSIONES_IMAGEN = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}
TIPOS_IMAGEN       = ["jpg", "jpeg", "png", "tiff", "bmp", "webp"]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background-color: #0f1117; color: #e8eaf0; }
section[data-testid="stSidebar"] { background-color: #161b27; border-right: 1px solid #1e2535; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.titulo-app {
    font-family: 'IBM Plex Mono', monospace; font-size: 1.85rem; font-weight: 600;
    color: #e8eaf0; letter-spacing: -0.5px; margin-bottom: 0.15rem;
}
.subtitulo-app {
    font-size: 0.78rem; color: #6b7280; letter-spacing: 0.1em;
    text-transform: uppercase; margin-bottom: 2rem;
    font-family: 'IBM Plex Mono', monospace;
}
.result-card {
    background: linear-gradient(135deg, #161b27 0%, #1a2035 100%);
    border: 1px solid #252d42; border-radius: 12px;
    padding: 1.8rem 2rem; margin: 1rem 0;
}
.conf-bar-track { background: #1e2535; border-radius: 3px; height: 7px; overflow: hidden; }
.conf-bar-fill  { height: 100%; border-radius: 3px; }
.sep { border: none; border-top: 1px solid #1e2535; margin: 1.5rem 0; }
.warn-box {
    background: #2d1f07; border: 1px solid #f59e0b; border-radius: 8px;
    padding: 0.8rem 1rem; font-size: 0.82rem; color: #fbbf24; margin: 0.5rem 0;
}
.info-box {
    background: #0d2137; border: 1px solid #2980b9; border-radius: 8px;
    padding: 0.8rem 1rem; font-size: 0.82rem; color: #7ec8e3; margin: 0.5rem 0;
}
.ok-box {
    background: #0d2218; border: 1px solid #27ae60; border-radius: 8px;
    padding: 0.8rem 1rem; font-size: 0.82rem; color: #6ee7a0; margin: 0.5rem 0;
}
.ocr-box {
    background: #1a1030; border: 1px solid #8e44ad; border-radius: 8px;
    padding: 0.8rem 1rem; font-size: 0.82rem; color: #c084fc; margin: 0.5rem 0;
}
.mover-card {
    background: #0f1623; border: 1px solid #1e2d45;
    border-radius: 10px; padding: 1rem 1.2rem; margin-top: 0.6rem;
}
.clase-dot {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    margin-right: 6px; vertical-align: middle;
}
.res-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 4px 0; border-bottom: 1px solid #1e2535;
    font-size: 0.82rem; color: #9ca3af;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ─── Utilidades de resultados ─────────────────────────────────────────────────

def carpeta_clase(clase: str) -> Path:
    p = RESULTADOS_DIR / clase
    p.mkdir(parents=True, exist_ok=True)
    return p


def guardar_en_resultados(nombre_archivo: str, contenido: bytes, clase: str) -> Path:
    destino_dir = carpeta_clase(clase)
    stem    = Path(nombre_archivo).stem
    sufijo  = Path(nombre_archivo).suffix or ".pdf"
    destino = destino_dir / f"{stem}{sufijo}"
    contador = 1
    while destino.exists():
        destino  = destino_dir / f"{stem}_{contador}{sufijo}"
        contador += 1
    destino.write_bytes(contenido)
    return destino


def conteo_resultados() -> dict:
    total = {}
    for c in COLORES:
        if (RESULTADOS_DIR / c).exists():
            pdfs = len(list((RESULTADOS_DIR / c).glob("*.pdf")))
            imgs = sum(len(list((RESULTADOS_DIR / c).glob(f"*.{ext}")))
                       for ext in TIPOS_IMAGEN)
            total[c] = pdfs + imgs
        else:
            total[c] = 0
    return total


# ─── Extracción de texto ──────────────────────────────────────────────────────

def extraer_desde_bytes(data: bytes, nombre: str) -> tuple[str, str, bool]:
    """
    Extrae texto crudo de PDF o imagen.
    Retorna (texto_raw, metodo_usado, es_imagen).
    """
    ext = Path(nombre).suffix.lower()

    if ext in EXTENSIONES_IMAGEN:
        # Ruta imagen → OCR
        from src.extractor import extraer_texto_imagen_bytes, TESSERACT_AVAILABLE
        if not TESSERACT_AVAILABLE:
            return "", "ocr_no_disponible", True
        texto = extraer_texto_imagen_bytes(data, nombre)
        return texto, "ocr_imagen", True
    else:
        # Ruta PDF → PyMuPDF
        import fitz
        try:
            doc      = fitz.open(stream=data, filetype="pdf")
            texto    = "".join(p.get_text("text") for p in doc)
            n_pags   = len(doc)
            doc.close()
            metodo = "pdf_digital"
            # Si el PDF parece escaneado, intentar OCR
            if len(texto.strip()) < 50:
                from src.extractor import TESSERACT_AVAILABLE
                if TESSERACT_AVAILABLE:
                    from src.extractor import extraer_texto_pdf
                    import tempfile, os
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                        tmp.write(data)
                        tmp_path = tmp.name
                    texto  = extraer_texto_pdf(tmp_path)
                    metodo = "ocr_pdf_escaneado"
                    os.unlink(tmp_path)
        except Exception as e:
            return "", f"error: {e}", False
        return texto, metodo, False


# ─── Carga de modelos ─────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def cargar_modelos():
    import joblib
    from config import NB_MODEL_PKL, SVM_MODEL_PKL
    if not NB_MODEL_PKL.exists() or not SVM_MODEL_PKL.exists():
        return None, None
    return joblib.load(NB_MODEL_PKL), joblib.load(SVM_MODEL_PKL)


def predecir(texto_limpio: str, pipeline_nb, pipeline_svm) -> dict:
    resultado = {}
    for nombre, pipeline in [("nb", pipeline_nb), ("svm", pipeline_svm)]:
        clase_pred = pipeline.predict([texto_limpio])[0]
        clf   = pipeline.named_steps["clf"]
        tfidf = pipeline.named_steps["tfidf"]
        X_vec = tfidf.transform([texto_limpio])
        if hasattr(clf, "decision_function"):
            scores = clf.decision_function(X_vec)[0]
            e      = np.exp(scores - scores.max())
            probs  = e / e.sum()
        else:
            probs = clf.predict_proba(X_vec)[0]
        clases = pipeline.classes_
        resultado[nombre] = {
            "clase":     clase_pred,
            "confianza": float(probs.max()),
            "scores":    {c: float(p) for c, p in zip(clases, probs)},
        }
    return resultado


def diagnostico_vocabulario(texto_limpio: str, pipeline) -> dict:
    vocab     = set(pipeline.named_steps["tfidf"].vocabulary_.keys())
    tokens    = texto_limpio.split()
    en_vocab  = [t for t in tokens if t in vocab]
    fuera     = [t for t in tokens if t not in vocab]
    cobertura = len(en_vocab) / max(len(tokens), 1)
    return {
        "total_tokens": len(tokens),
        "en_vocab":     len(en_vocab),
        "fuera_vocab":  len(fuera),
        "cobertura":    cobertura,
        "tokens_fuera": fuera[:20],
        "tokens_clave": en_vocab[:20],
    }


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Configuración")
    modelo_activo       = st.radio("Modelo principal",
                                   ["SVM (recomendado)", "Naive Bayes", "Ambos"],
                                   index=0)
    mostrar_diagnostico = st.toggle("Diagnóstico de vocabulario", value=True)

    st.markdown("<hr class='sep'>", unsafe_allow_html=True)
    st.markdown("### Clases soportadas")
    for clase, desc in DESCRIPCIONES.items():
        color = COLORES[clase]
        st.markdown(
            f"<div style='margin-bottom:10px'>"
            f"<span style='color:{color};font-weight:700'>"
            f"<span class='clase-dot' style='background:{color}'></span>{clase.upper()}</span>"
            f"<br><span style='font-size:0.78rem;color:#6b7280'>{desc}</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr class='sep'>", unsafe_allow_html=True)
    st.markdown("### Rendimiento")
    perf = pd.DataFrame({
        "Modelo":   ["Naive Bayes", "SVM"],
        "Accuracy": ["92.3 %",      "95.8 %"],
        "F1 CV":    ["93.1 ±1.8 %", "96.2 ±1.1 %"],
    })
    st.dataframe(perf, hide_index=True, width="stretch")

    # ── Estado OCR ────────────────────────────────────────────────────────────
    st.markdown("<hr class='sep'>", unsafe_allow_html=True)
    st.markdown("### Estado OCR")
    try:
        from src.extractor import verificar_ocr
        ocr_estado = verificar_ocr()
        if ocr_estado.get("tesseract"):
            v = ocr_estado.get("tesseract_version", "")
            spa = "✓ spa" if ocr_estado.get("spa_disponible") else "⚠ spa falta"
            st.markdown(
                f"<div class='ok-box'>Tesseract {v}<br>{spa}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='warn-box'>Tesseract no instalado.<br>"
                "Solo se procesan PDFs con texto digital.<br>"
                "<code>pip install pytesseract Pillow</code></div>",
                unsafe_allow_html=True,
            )
    except Exception:
        st.caption("Estado OCR no disponible")

    # ── Estado de resultados/ ─────────────────────────────────────────────────
    st.markdown("<hr class='sep'>", unsafe_allow_html=True)
    st.markdown("### Carpeta de resultados")
    conteo = conteo_resultados()
    total  = sum(conteo.values())
    if total == 0:
        st.caption("Aún no hay documentos guardados.")
    else:
        for clase, n in conteo.items():
            if n == 0:
                continue
            color = COLORES[clase]
            st.markdown(
                f"<div class='res-row'>"
                f"<span><span class='clase-dot' style='background:{color}'></span>{clase}</span>"
                f"<span style='font-family:IBM Plex Mono,monospace;font-size:0.78rem'>"
                f"{n} archivo(s)</span></div>",
                unsafe_allow_html=True,
            )
        st.caption(f"Total: {total} · {RESULTADOS_DIR.name}/")

    if st.button("Abrir carpeta de resultados", width="stretch"):
        import subprocess, platform
        RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            if platform.system() == "Windows":
                subprocess.Popen(["explorer", str(RESULTADOS_DIR)])
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", str(RESULTADOS_DIR)])
            else:
                subprocess.Popen(["xdg-open", str(RESULTADOS_DIR)])
        except Exception:
            pass
        st.info(f"Ruta: `{RESULTADOS_DIR}`")

    st.markdown("<hr class='sep'>", unsafe_allow_html=True)
    st.caption("TF-IDF + Naive Bayes / SVM · OCR")


# ─── Encabezado ───────────────────────────────────────────────────────────────
st.markdown("<div class='titulo-app'>Clasificador de Documentos</div>",
            unsafe_allow_html=True)
st.markdown(
    "<div class='subtitulo-app'>Análisis automático · TF-IDF + Machine Learning · OCR</div>",
    unsafe_allow_html=True,
)

# ─── Verificar modelos ────────────────────────────────────────────────────────
pipeline_nb, pipeline_svm = cargar_modelos()
if pipeline_nb is None:
    st.error(
        "**Modelos no encontrados.** Ejecuta primero:\n"
        "```bash\npython main.py realistic\npython main.py extract\npython main.py train\n```"
    )
    st.stop()

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_clasificar, tab_lote, tab_anonimizar = st.tabs(
    ["Clasificar documento", "Clasificación por lote", "Anonimizar PDF real"]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CLASIFICACIÓN INDIVIDUAL
# ══════════════════════════════════════════════════════════════════════════════
with tab_clasificar:
    col_izq, col_der = st.columns([1, 1.15], gap="large")

    with col_izq:
        st.markdown("#### Subir documento")
        st.markdown(
            "<div class='info-box'>Acepta <b>PDF</b> (texto digital o escaneado) "
            "e <b>imágenes</b> de documentos (JPG, PNG, TIFF…). "
            "Las imágenes se procesan con OCR.</div>",
            unsafe_allow_html=True,
        )
        archivo = st.file_uploader(
            "Arrastra un archivo o haz clic para seleccionar",
            type=["pdf"] + TIPOS_IMAGEN,
            label_visibility="collapsed",
            key="single",
        )

        if archivo:
            from src.preprocessor import limpiar_texto

            if st.session_state.get("ind_nombre") != archivo.name:
                st.session_state["ind_file_bytes"] = archivo.read()
                st.session_state["ind_nombre"]     = archivo.name
                st.session_state.pop("ind_guardado", None)

            file_bytes = st.session_state["ind_file_bytes"]
            es_imagen  = Path(archivo.name).suffix.lower() in EXTENSIONES_IMAGEN

            # ── Metadatos y vista previa ───────────────────────────────────────
            st.markdown(f"**Archivo:** `{archivo.name}`")
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Tamaño", f"{archivo.size / 1024:.1f} KB")
            col_m2.metric("Tipo", "Imagen" if es_imagen else "PDF")

            if es_imagen:
                # Vista previa de la imagen
                st.image(file_bytes, caption="Vista previa", width="stretch")
                st.markdown(
                    "<div class='ocr-box'>🔍 Imagen detectada — se aplicará OCR "
                    "con Tesseract para extraer el texto antes de clasificar.</div>",
                    unsafe_allow_html=True,
                )
            else:
                import fitz
                try:
                    doc       = fitz.open(stream=file_bytes, filetype="pdf")
                    n_paginas = len(doc)
                    doc.close()
                    col_m3.metric("Páginas", n_paginas)
                except Exception:
                    col_m3.metric("Páginas", "—")

            # ── Extraer texto ──────────────────────────────────────────────────
            with st.spinner("Extrayendo texto…"):
                texto_raw, metodo, _ = extraer_desde_bytes(file_bytes, archivo.name)
                texto_lim = limpiar_texto(texto_raw)

            col_m3.metric("Tokens", len(texto_lim.split()))

            if metodo == "ocr_no_disponible":
                st.error(
                    "**Tesseract no instalado.** Para procesar imágenes:\n"
                    "```bash\npip install pytesseract Pillow\n```\n"
                    "Luego instala Tesseract desde https://github.com/UB-Mannheim/tesseract/wiki"
                )
            elif len(texto_lim.split()) < 10:
                st.markdown(
                    "<div class='warn-box'><b>Advertencia:</b> Muy poco texto extraído. "
                    + ("Verifica que la imagen sea legible y el idioma sea español."
                       if es_imagen else
                       "El PDF podría estar escaneado. Activa Tesseract OCR.")
                    + "</div>",
                    unsafe_allow_html=True,
                )
            else:
                metodo_label = {
                    "pdf_digital":       "PDF digital",
                    "ocr_imagen":        "OCR imagen",
                    "ocr_pdf_escaneado": "OCR PDF escaneado",
                }.get(metodo, metodo)
                st.caption(f"Método de extracción: {metodo_label}")

            st.markdown("<hr class='sep'>", unsafe_allow_html=True)
            if texto_raw and not es_imagen:
                st.markdown("**Vista previa del texto:**")
                preview = texto_raw.strip()[:500].replace("<", "&lt;").replace(">", "&gt;")
                st.markdown(
                    f"<div style='background:#0d1117;border:1px solid #1e2535;"
                    f"border-radius:8px;padding:0.8rem;font-family:IBM Plex Mono,monospace;"
                    f"font-size:0.78rem;color:#6b7280;max-height:180px;overflow-y:auto'>"
                    f"{preview}…</div>",
                    unsafe_allow_html=True,
                )
            elif texto_raw and es_imagen:
                with st.expander("Texto extraído por OCR"):
                    st.text(texto_raw[:800])

    with col_der:
        st.markdown("#### Resultado de clasificación")

        if not archivo or metodo == "ocr_no_disponible":
            st.markdown(
                "<div style='background:#161b27;border:1px dashed #252d42;"
                "border-radius:12px;padding:3rem 2rem;text-align:center;color:#374151'>"
                "<div style='font-family:IBM Plex Mono,monospace;font-size:1.6rem;"
                "margin-bottom:1rem;color:#252d42'>◈</div>"
                "<div style='font-family:IBM Plex Mono,monospace;font-size:0.85rem'>"
                "Sube un PDF o imagen para clasificarlo</div></div>",
                unsafe_allow_html=True,
            )
        elif texto_lim.strip():
            with st.spinner("Clasificando…"):
                time.sleep(0.15)
                res = predecir(texto_lim, pipeline_nb, pipeline_svm)

            modelo_ppal = "svm" if "Naive" not in modelo_activo else "nb"
            clase_pred  = res[modelo_ppal]["clase"]
            confianza   = res[modelo_ppal]["confianza"]
            scores      = res[modelo_ppal]["scores"]
            color       = COLORES.get(clase_pred, "#888")

            # Resultado principal
            st.markdown(
                f"<div class='result-card'>"
                f"<div style='display:inline-block;font-family:IBM Plex Mono,monospace;"
                f"font-size:0.72rem;font-weight:600;letter-spacing:0.12em;"
                f"text-transform:uppercase;padding:4px 12px;border-radius:4px;"
                f"background:{color}22;color:{color};border:1px solid {color}55;"
                f"margin-bottom:8px'>"
                f"<span class='clase-dot' style='background:{color}'></span>"
                f"{clase_pred.upper()}</div>"
                f"<div style='font-size:1.9rem;font-weight:700;color:{color};"
                f"font-family:IBM Plex Mono,monospace;margin-bottom:4px'>"
                f"{confianza:.1%}</div>"
                f"<div style='font-size:0.82rem;color:#9ca3af;margin-bottom:1.2rem'>"
                f"{DESCRIPCIONES.get(clase_pred, '')}</div>",
                unsafe_allow_html=True,
            )
            for clase_i, score_i in sorted(scores.items(), key=lambda x: -x[1]):
                col_i    = COLORES.get(clase_i, "#888")
                pct      = score_i * 100
                negrita  = "font-weight:700;" if clase_i == clase_pred else ""
                lcolor   = col_i if clase_i == clase_pred else "#9ca3af"
                opacidad = "1" if clase_i == clase_pred else "0.4"
                st.markdown(
                    f"<div style='margin:7px 0'>"
                    f"<div style='display:flex;justify-content:space-between;"
                    f"font-size:10.5px;color:#6b7280;margin-bottom:3px'>"
                    f"<span style='{negrita}color:{lcolor}'>"
                    f"<span class='clase-dot' style='background:{col_i};"
                    f"opacity:{opacidad}'></span>{clase_i}</span>"
                    f"<span style='font-family:IBM Plex Mono,monospace;"
                    f"{negrita}color:{lcolor}'>{pct:.1f}%</span></div>"
                    f"<div class='conf-bar-track'>"
                    f"<div class='conf-bar-fill' style='width:{pct:.1f}%;"
                    f"background:{col_i};opacity:{opacidad}'></div></div></div>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

            # Guardar en resultados
            st.markdown("<hr class='sep'>", unsafe_allow_html=True)
            if "ind_guardado" in st.session_state:
                ruta_ok = Path(st.session_state["ind_guardado"])
                if str(ruta_ok) != "omitido":
                    st.markdown(
                        f"<div class='ok-box'>Guardado en "
                        f"<code>resultados/{ruta_ok.parent.name}/{ruta_ok.name}</code></div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    f"<div class='mover-card'>"
                    f"<div style='font-size:0.85rem;color:#cbd5e1;margin-bottom:0.8rem'>"
                    f"Clasificado como <b style='color:{color}'>{clase_pred.upper()}</b>. "
                    f"¿Mover a <code>resultados/{clase_pred}/</code>?</div>",
                    unsafe_allow_html=True,
                )
                col_sel, col_si, col_no = st.columns([1.6, 0.8, 0.8], gap="small")
                with col_sel:
                    clase_destino = st.selectbox(
                        "Carpeta destino", list(COLORES.keys()),
                        index=list(COLORES.keys()).index(clase_pred),
                        key="clase_destino_ind", label_visibility="collapsed",
                    )
                with col_si:
                    if st.button("Mover", type="primary",
                                 width="stretch", key="btn_mover"):
                        ruta_ok = guardar_en_resultados(
                            archivo.name, file_bytes, clase_destino
                        )
                        st.session_state["ind_guardado"] = str(ruta_ok)
                        st.rerun()
                with col_no:
                    if st.button("Omitir", width="stretch", key="btn_omitir"):
                        st.session_state["ind_guardado"] = "omitido"
                        st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

            # Comparativa NB vs SVM
            st.markdown("<hr class='sep'>", unsafe_allow_html=True)
            st.markdown("**Comparativa NB vs SVM:**")
            col_nb, col_svm_c = st.columns(2)
            for col_v, clave, label in [
                (col_nb,    "nb",  "Naive Bayes"),
                (col_svm_c, "svm", "SVM"),
            ]:
                with col_v:
                    cls_v = res[clave]["clase"]
                    c     = COLORES.get(cls_v, "#888")
                    st.markdown(
                        f"<div style='background:#161b27;border:1px solid #1e2535;"
                        f"border-radius:8px;padding:10px;text-align:center'>"
                        f"<div style='font-size:10px;color:#6b7280;"
                        f"font-family:IBM Plex Mono,monospace'>{label}</div>"
                        f"<div style='font-weight:700;color:{c};margin:4px 0'>"
                        f"<span class='clase-dot' style='background:{c}'></span>"
                        f"{cls_v.upper()}</div>"
                        f"<div style='font-size:10px;color:#6b7280'>"
                        f"{res[clave]['confianza']:.1%}</div></div>",
                        unsafe_allow_html=True,
                    )
            if res["nb"]["clase"] != res["svm"]["clase"]:
                st.markdown(
                    "<div class='warn-box'><b>Los modelos discrepan.</b> "
                    "El documento puede ser ambiguo. Revisa el diagnóstico.</div>",
                    unsafe_allow_html=True,
                )

            # Diagnóstico de vocabulario
            if mostrar_diagnostico:
                st.markdown("<hr class='sep'>", unsafe_allow_html=True)
                st.markdown("**Diagnóstico de vocabulario:**")
                diag      = diagnostico_vocabulario(texto_lim, pipeline_svm)
                cobertura = diag["cobertura"]
                col_d1, col_d2, col_d3 = st.columns(3)
                col_d1.metric("Tokens totales",  diag["total_tokens"])
                col_d2.metric("En vocabulario",  f"{diag['en_vocab']} ({cobertura:.0%})")
                col_d3.metric("Fuera del vocab", diag["fuera_vocab"])
                if cobertura < 0.40:
                    st.markdown(
                        f"<div class='warn-box'><b>Cobertura baja ({cobertura:.0%}).</b> "
                        "Considera reentrenar con documentos similares.</div>",
                        unsafe_allow_html=True,
                    )
                elif cobertura < 0.65:
                    st.markdown(
                        f"<div class='warn-box'><b>Cobertura moderada ({cobertura:.0%}).</b> "
                        "Algunos términos no están en el vocabulario.</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"<div class='ok-box'><b>Buena cobertura ({cobertura:.0%}).</b></div>",
                        unsafe_allow_html=True,
                    )
                if diag["tokens_fuera"]:
                    with st.expander(f"Tokens no reconocidos ({len(diag['tokens_fuera'])} mostrados)"):
                        st.code(" · ".join(diag["tokens_fuera"]), language=None)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CLASIFICACIÓN POR LOTE
# ══════════════════════════════════════════════════════════════════════════════
with tab_lote:
    st.markdown("#### Clasificar múltiples archivos")
    st.markdown(
        "<div class='info-box'>Acepta PDFs e imágenes mezclados en el mismo lote.</div>",
        unsafe_allow_html=True,
    )
    archivos_lote = st.file_uploader(
        "Sube varios archivos",
        type=["pdf"] + TIPOS_IMAGEN,
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="lote",
    )

    if archivos_lote:
        st.markdown(f"**{len(archivos_lote)} archivo(s) cargado(s)**")
        if st.button("Clasificar todos", type="primary"):
            from src.preprocessor import limpiar_texto

            filas      = []
            contenidos = {}
            barra      = st.progress(0, text="Procesando…")

            for i, f in enumerate(archivos_lote):
                raw                   = f.read()
                texto_raw, metodo, _  = extraer_desde_bytes(raw, f.name)
                lim                   = limpiar_texto(texto_raw)
                r                     = predecir(lim, pipeline_nb, pipeline_svm)
                diag                  = diagnostico_vocabulario(lim, pipeline_svm)
                contenidos[f.name]    = (raw, r["svm"]["clase"])
                filas.append({
                    "Archivo":         f.name,
                    "Tipo":            "Imagen" if Path(f.name).suffix.lower() in EXTENSIONES_IMAGEN else "PDF",
                    "Método":          metodo,
                    "Clase (SVM)":     r["svm"]["clase"],
                    "Confianza (SVM)": f"{r['svm']['confianza']:.1%}",
                    "Clase (NB)":      r["nb"]["clase"],
                    "Coinciden":       "si" if r["svm"]["clase"] == r["nb"]["clase"] else "no",
                    "Cobertura":       f"{diag['cobertura']:.0%}",
                })
                barra.progress((i + 1) / len(archivos_lote),
                               text=f"Procesando {i+1}/{len(archivos_lote)}…")

            barra.empty()
            st.session_state["df_lote"]    = filas
            st.session_state["contenidos"] = contenidos
            st.session_state.pop("lote_guardados", None)

    if "df_lote" in st.session_state:
        df_lote    = pd.DataFrame(st.session_state["df_lote"])
        contenidos = st.session_state.get("contenidos", {})

        st.dataframe(df_lote, width="stretch", hide_index=True)

        col_csv, col_org = st.columns(2, gap="small")
        with col_csv:
            st.download_button(
                "Descargar CSV",
                data=df_lote.to_csv(index=False).encode("utf-8"),
                file_name="clasificacion_lote.csv",
                mime="text/csv",
                width="stretch",
            )

        if "lote_guardados" not in st.session_state:
            with col_org:
                if st.button("Mover todos a resultados/", type="primary",
                             width="stretch", key="btn_mover_lote"):
                    guardados = {
                        nombre: str(guardar_en_resultados(nombre, raw, clase))
                        for nombre, (raw, clase) in contenidos.items()
                    }
                    st.session_state["lote_guardados"] = guardados
                    st.rerun()

        # Distribución de clases
        st.markdown("<hr class='sep'>", unsafe_allow_html=True)
        st.markdown("**Distribución de clases (SVM):**")
        dist = df_lote["Clase (SVM)"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 2.5), facecolor="#161b27")
        ax.set_facecolor("#161b27")
        ax.barh(dist.index, dist.values,
                color=[COLORES.get(c, "#888") for c in dist.index], alpha=0.85)
        ax.tick_params(colors="#9ca3af")
        ax.set_xlabel("Documentos", color="#9ca3af")
        for sp in ax.spines.values():
            sp.set_edgecolor("#252d42")
        for idx_b, v in enumerate(dist.values):
            ax.text(v + 0.05, idx_b, str(v), va="center", color="#e8eaf0", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ANONIMIZADOR
# ══════════════════════════════════════════════════════════════════════════════
with tab_anonimizar:
    st.markdown("#### Anonimizar PDFs reales para entrenamiento")
    st.markdown(
        "<div class='info-box'>Elimina información sensible (RUTs, nombres, montos, "
        "fechas, correos, teléfonos) antes de usar documentos reales como datos de "
        "entrenamiento. Los originales no se modifican.</div>",
        unsafe_allow_html=True,
    )

    col_a1, col_a2 = st.columns([1, 1])
    with col_a1:
        clase_real   = st.selectbox("Clase documental",
                                    ["factura", "contrato", "informe", "cotizacion", "orden_compra"])
        archivo_real = st.file_uploader(
            "Sube el PDF original",
            type=["pdf"], key="anon_upload", label_visibility="collapsed",
        )

    with col_a2:
        if archivo_real:
            import fitz
            from src.anonymizer import anonimizar_texto

            doc        = fitz.open(stream=archivo_real.read(), filetype="pdf")
            texto_orig = "".join(p.get_text("text") for p in doc)
            doc.close()

            texto_anon, stats = anonimizar_texto(texto_orig)
            total_reem        = sum(stats.values())

            st.markdown("**Resultado de anonimización:**")
            for token, count in stats.items():
                st.markdown(
                    f"<div style='font-family:IBM Plex Mono,monospace;"
                    f"font-size:0.8rem;color:#9ca3af'>{token}: "
                    f"<b style='color:#e8eaf0'>{count}</b></div>",
                    unsafe_allow_html=True,
                )
            st.markdown(f"**Total: {total_reem} elemento(s) anonimizado(s)**")

            with st.expander("Original vs anonimizado (primeros 500 chars)"):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Original:**")
                    st.code(texto_orig[:500], language=None)
                with c2:
                    st.markdown("**Anonimizado:**")
                    st.code(texto_anon[:500], language=None)

            if st.button("Descargar PDF anonimizado", type="primary"):
                from reportlab.lib.pagesizes import A4
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
                from reportlab.lib.styles import ParagraphStyle
                from reportlab.lib.units import cm

                buf     = io.BytesIO()
                doc_out = SimpleDocTemplate(
                    buf, pagesize=A4,
                    leftMargin=2.5*cm, rightMargin=2.5*cm,
                    topMargin=2*cm, bottomMargin=2*cm,
                )
                estilo = ParagraphStyle("b", fontSize=9, leading=14, spaceAfter=6)
                story  = []
                for linea in texto_anon.split("\n"):
                    linea = linea.strip()
                    if linea:
                        linea = (linea.replace("&", "&amp;")
                                      .replace("<", "&lt;")
                                      .replace(">", "&gt;"))
                        try:
                            story.append(Paragraph(linea, estilo))
                        except Exception:
                            pass
                    else:
                        story.append(Spacer(1, 0.2*cm))
                doc_out.build(story)

                nombre_salida = f"{Path(archivo_real.name).stem}_anonimizado.pdf"
                st.download_button(
                    nombre_salida,
                    data=buf.getvalue(),
                    file_name=nombre_salida,
                    mime="application/pdf",
                )
                st.markdown(
                    f"<div class='ok-box'>Descarga lista. Para agregar al corpus:<br>"
                    f"<code>python main.py anonymize --input data/real/{clase_real}/ "
                    f"--clase {clase_real}</code><br>"
                    f"<code>python main.py extract &amp;&amp; "
                    f"python main.py train</code></div>",
                    unsafe_allow_html=True,
                )
