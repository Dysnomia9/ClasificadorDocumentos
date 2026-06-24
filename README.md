# Clasificador Automático de Documentos Empresariales

Clasifica documentos PDF e imágenes en **cinco categorías** del ecosistema empresarial
chileno, usando TF-IDF + Naive Bayes y SVM Lineal. Incluye OCR para imágenes de documentos,
interfaz web Streamlit y evaluación con documentos reales.

---

## Clases soportadas

| Clase | Descripción | Vocabulario clave |
|-------|-------------|-------------------|
| `factura` | Factura / boleta electrónica | RUT, SII, IVA, folio, neto, timbre electrónico |
| `contrato` | Contratos de trabajo, arriendo, servicios | cláusula, arrendatario, Código del Trabajo, vigencia |
| `informe` | Informes de gestión y reportes técnicos | KPI, indicadores, resumen ejecutivo, período evaluado |
| `cotizacion` | Cotizaciones y presupuestos comerciales | validez, sin compromiso, valores más IVA, ejecutivo comercial |
| `orden_compra` | Órdenes de compra (OC) | bodega, centro de costo, recepción conforme, despacho |

---

## Inicio rápido

```bash
pip install -r requirements.txt
python main.py realistic          # genera corpus (145 docs × 5 clases)
python main.py extract            # extrae texto con PyMuPDF + OCR
python main.py train              # entrena NB y SVM con GridSearch
python main.py evaluate           # genera figuras y reporte
streamlit run app.py              # interfaz web
```

> **Reproducibilidad:** `python main.py realistic` usa `--n 145` por defecto
> (725 documentos = 145 × 5 clases) y una semilla fija (`RANDOM_STATE` en `config.py`).
> Los PDFs, el dataset y los modelos no se versionan: se regeneran con los comandos de arriba.

---

## Comandos disponibles

| Comando | Descripción |
|---------|-------------|
| `realistic` | Corpus realista 5 clases — pools train/test disjuntos **(recomendado)** |
| `generate` | Dataset sintético básico (vocabulario más acotado) |
| `anonymize` | Anonimiza PDFs reales: elimina RUTs, nombres, montos, fechas |
| `noise` | Inyecta ruido semántico al corpus de entrenamiento |
| `extract` | Extrae texto de PDFs (PyMuPDF) e imágenes (Tesseract OCR) |
| `train` | Entrena NB y SVM con GridSearchCV y validación cruzada |
| `evaluate` | Matrices de confusión, curvas de aprendizaje, reporte |
| `features` | Análisis TF-IDF: top palabras por clase y heatmap |
| `predict <ruta>` | Clasifica un PDF o imagen nueva |
| `pipeline` | Ejecuta la secuencia completa |
| `status` | Muestra estado actual del proyecto |

---

## Evaluación con documentos reales

Para métricas sobre documentos reales (no sintéticos):

```
docs_reales/
├── factura/       ← PDFs o fotos de facturas reales
├── contrato/
├── informe/
├── cotizacion/
└── orden_compra/
```

```bash
python src/eval_real.py --input docs_reales/
```

Genera en `reports/evaluacion_real/`: matrices de confusión, comparativa de métricas,
F1 por clase, distribución de confianza y reporte completo.

---

## OCR para imágenes

La app acepta imágenes (JPG, PNG, TIFF, BMP, WEBP) además de PDFs.
El pipeline preprocesa la imagen antes del OCR:

1. Conversión a escala de grises
2. Reescalado mínimo 1 800 px de ancho
3. Realce de contraste
4. Corrección de orientación EXIF (fotos de celular)
5. Tesseract con idioma español (`spa`)

Instalar Tesseract:
```bash
# Linux
sudo apt install tesseract-ocr tesseract-ocr-spa

# macOS
brew install tesseract tesseract-lang

# Windows
# Descargar instalador: https://github.com/UB-Mannheim/tesseract/wiki
# Agregar al PATH y descargar spa.traineddata
```

---

## Anonimizador

| Patrón | Token |
|--------|-------|
| RUTs chilenos (76.123.456-7) | `[RUT]` |
| Nombres propios (mínimo 3 palabras) | `[NOMBRE_PROPIO]` |
| Nombres en contexto formal (don/doña) | `[NOMBRE]` |
| Montos en pesos ($850.000) | `[MONTO]` |
| Fechas | `[FECHA]` |
| Correos electrónicos | `[CORREO]` |
| Teléfonos chilenos (+56 9 XXXX) | `[TELÉFONO]` |
| Folios (N°1234) | `N°[FOLIO]` |
| Cuentas bancarias | `[NÚMERO]` |
| Direcciones (Av./Calle + número) | `[DIRECCIÓN]` |

---

## Dependencias principales

```
pymupdf, reportlab, scikit-learn, nltk, pandas, numpy,
matplotlib, seaborn, joblib, streamlit, Pillow, pytesseract
```
