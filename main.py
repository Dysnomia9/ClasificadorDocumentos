"""
Clasificador Automático de Documentos Empresariales
====================================================
Proyecto personal

5 clases: factura · contrato · informe · cotizacion · orden_compra

Comandos:
  python main.py generate          → Dataset sintético básico
  python main.py realistic         → Corpus realista con 5 clases (recomendado)
  python main.py anonymize         → Anonimiza PDFs reales
  python main.py noise             → Inyecta ruido semántico al corpus
  python main.py extract           → Extrae texto de PDFs e imágenes (OCR)
  python main.py train             → Entrena NB y SVM
  python main.py evaluate          → Genera gráficos y reporte
  python main.py features          → Análisis TF-IDF top palabras
  python main.py predict <ruta>    → Clasifica PDF o imagen nueva
  python main.py pipeline          → Pipeline completo
  python main.py status            → Estado del proyecto
"""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))


def cmd_generate(args):
    print("\n" + "═"*55)
    print("  GENERANDO DATASET SINTÉTICO (5 CLASES)")
    print("═"*55)
    from src.generate_dataset import generar_dataset
    from config import DOCS_POR_CLASE
    n  = args.n if hasattr(args,"n") and args.n else DOCS_POR_CLASE
    df = generar_dataset(n_por_clase=n, verbose=True)
    print(f"\n  Total documentos: {len(df)}")


def cmd_realistic(args):
    print("\n" + "═"*55)
    print("  GENERANDO CORPUS REALISTA (5 CLASES)")
    print("═"*55)
    from src.realistic_corpus import regenerar_corpus_realista
    n = args.n if hasattr(args,"n") and args.n else 145
    df = regenerar_corpus_realista(n_por_clase=n, frac_test=0.20,
                                   frac_mixtos=0.38, frac_cortos=0.10, verbose=True)
    print(f"\n  ✓ Corpus generado: {len(df)} documentos")
    print("  → Ejecuta: python main.py extract && python main.py train")


def cmd_anonymize(args):
    print("\n" + "═"*55)
    print("  ANONIMIZACIÓN DE DOCUMENTOS REALES")
    print("═"*55)
    if not hasattr(args,"input") or not args.input:
        print("[ERROR] Uso: python main.py anonymize --input <carpeta_o_pdf> --clase <clase>")
        print("  Clases: factura, contrato, informe, cotizacion, orden_compra")
        return
    from src.anonymizer import procesar_carpeta, anonimizar_pdf
    ruta = Path(args.input)
    if ruta.is_dir():
        registros = procesar_carpeta(ruta, args.clase, verbose=True)
        print(f"\n  ✓ {len(registros)} documentos anonimizados")
    elif ruta.is_file():
        from config import DATA_RAW_DIR
        salida = DATA_RAW_DIR / args.clase / f"real_{ruta.stem}_anon.pdf"
        salida.parent.mkdir(parents=True, exist_ok=True)
        stats = anonimizar_pdf(ruta, salida)
        print(f"  ✓ {salida.name}: {stats}")
    else:
        print(f"[ERROR] {ruta} no existe")
    print("  → Ejecuta: python main.py extract && python main.py train")


def cmd_noise(args):
    print("\n" + "═"*55)
    print("  INYECCIÓN DE RUIDO SEMÁNTICO")
    print("═"*55)
    from src.noise_injector import inyectar_ruido_dataset
    frac = args.frac if hasattr(args,"frac") and args.frac else 0.25
    df   = inyectar_ruido_dataset(fraccion_ruidosos=frac, verbose=True)
    print(f"\n  Documentos ruidosos: {df['ruidoso'].sum()} ({frac:.0%})")


def cmd_extract(args):
    print("\n" + "═"*55)
    print("  EXTRACCIÓN DE TEXTO (PDF + OCR IMÁGENES)")
    print("═"*55)
    import pandas as pd
    from src.extractor import extraer_corpus, verificar_ocr
    from config import DATASET_CSV

    print("  Estado OCR:")
    ocr = verificar_ocr()
    for k,v in ocr.items():
        print(f"    {'✓' if v is True else ('✗' if v is False else '→')} {k}: {v}")

    if not DATASET_CSV.exists():
        print("[ERROR] Ejecuta primero: python main.py realistic")
        sys.exit(1)
    df = pd.read_csv(DATASET_CSV)
    if "texto" in df.columns:
        df.loc[df["texto"].isna(), "texto"] = None
    df = extraer_corpus(df, verbose=True)
    print(f"\n  Longitud media: {df['texto'].str.len().mean():.0f} chars")


def cmd_train(args):
    print("\n" + "═"*55)
    print("  ENTRENAMIENTO NB vs SVM")
    print("═"*55)
    import pandas as pd
    from src.extractor import extraer_corpus
    from src.preprocessor import preprocesar_corpus
    from src.trainer import entrenar
    from config import DATASET_CSV
    if not DATASET_CSV.exists():
        print("[ERROR] Ejecuta primero: python main.py realistic")
        sys.exit(1)
    df = pd.read_csv(DATASET_CSV)
    if "texto" not in df.columns or df["texto"].isna().any():
        df = extraer_corpus(df, verbose=True)
    print("\n  → Preprocesando texto...")
    df = preprocesar_corpus(df)
    grid = not (hasattr(args,"no_grid") and args.no_grid)
    resultados = entrenar(df, usar_grid_search=grid, verbose=True)
    import joblib
    from config import MODELS_DIR
    joblib.dump(resultados, MODELS_DIR / "resultados_entrenamiento.pkl")
    print("\n" + "─"*55)
    print(f"  NB:  Acc={resultados['acc_nb']:.4f}  F1-CV={resultados['cv_nb'].mean():.4f}±{resultados['cv_nb'].std():.4f}")
    print(f"  SVM: Acc={resultados['acc_svm']:.4f}  F1-CV={resultados['cv_svm'].mean():.4f}±{resultados['cv_svm'].std():.4f}")


def cmd_evaluate(args):
    print("\n" + "═"*55)
    print("  EVALUACIÓN Y VISUALIZACIONES")
    print("═"*55)
    import joblib
    from src.evaluator import evaluar_completo
    from config import MODELS_DIR, REPORT_TXT
    pkl = MODELS_DIR / "resultados_entrenamiento.pkl"
    if not pkl.exists():
        print("[ERROR] Ejecuta primero: python main.py train")
        sys.exit(1)
    resultados = joblib.load(pkl)
    evaluar_completo(resultados, verbose=True)
    print("\n" + REPORT_TXT.read_text(encoding="utf-8"))


def cmd_features(args):
    print("\n" + "═"*55)
    print("  ANÁLISIS DE FEATURES TF-IDF")
    print("═"*55)
    from src.feature_analysis import analizar_features
    analizar_features(verbose=True)


def cmd_predict(args):
    if not hasattr(args,"pdf") or not args.pdf:
        print("[ERROR] Uso: python main.py predict <ruta.pdf o ruta.jpg>")
        sys.exit(1)
    modelo = args.modelo if hasattr(args,"modelo") and args.modelo else "ambos"
    from src.predictor import clasificar_documento
    clasificar_documento(args.pdf, modelo=modelo, verbose=True)


def cmd_pipeline(args):
    print("\n" + "═"*55 + "\n  PIPELINE COMPLETO\n" + "═"*55)
    cmd_realistic(args)
    cmd_extract(args)
    cmd_train(args)
    cmd_evaluate(args)
    cmd_features(args)
    print("\n" + "═"*55 + "\n  ✓ PIPELINE COMPLETADO\n" + "═"*55)


def cmd_status(args):
    from config import DATA_RAW_DIR, DATASET_CSV, NB_MODEL_PKL, SVM_MODEL_PKL, REPORTS_DIR, CLASES
    print("\n" + "═"*55 + "\n  ESTADO DEL PROYECTO\n" + "═"*55)
    total = 0
    for clase in CLASES:
        carpeta = DATA_RAW_DIR / clase
        n = len(list(carpeta.glob("*.pdf"))) if carpeta.exists() else 0
        total += n
        print(f"  {'✓' if n>0 else '✗'} {clase:14s}: {n} PDFs")
    print(f"  Total PDFs : {total}")
    print(f"  Dataset CSV: {'✓' if DATASET_CSV.exists() else '✗'}")
    print(f"  Modelo NB  : {'✓' if NB_MODEL_PKL.exists() else '✗'}")
    print(f"  Modelo SVM : {'✓' if SVM_MODEL_PKL.exists() else '✗'}")
    if REPORTS_DIR.exists():
        print(f"  Reportes   : {len(list(REPORTS_DIR.iterdir()))} archivos → {REPORTS_DIR}")


def main():
    parser = argparse.ArgumentParser(
        description="Clasificador de Documentos Empresariales — 5 clases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="comando")

    p = sub.add_parser("generate"); p.add_argument("--n", type=int, default=None)
    p = sub.add_parser("realistic"); p.add_argument("--n", type=int, default=145)
    p = sub.add_parser("anonymize")
    p.add_argument("--input","-i", default=None)
    p.add_argument("--clase","-c", choices=["factura","contrato","informe","cotizacion","orden_compra"], default="contrato")
    p = sub.add_parser("noise"); p.add_argument("--frac", type=float, default=0.25)
    sub.add_parser("extract")
    p = sub.add_parser("train"); p.add_argument("--no-grid", dest="no_grid", action="store_true")
    sub.add_parser("evaluate")
    sub.add_parser("features")
    p = sub.add_parser("predict")
    p.add_argument("pdf")
    p.add_argument("--modelo", choices=["nb","svm","ambos"], default="ambos")
    p = sub.add_parser("pipeline")
    p.add_argument("--n", type=int, default=145)
    p.add_argument("--no-grid", dest="no_grid", action="store_true")
    sub.add_parser("status")

    args = parser.parse_args()
    cmds = {
        "generate": cmd_generate, "realistic":  cmd_realistic,
        "anonymize":cmd_anonymize,"noise":       cmd_noise,
        "extract":  cmd_extract,  "train":       cmd_train,
        "evaluate": cmd_evaluate, "features":    cmd_features,
        "predict":  cmd_predict,  "pipeline":    cmd_pipeline,
        "status":   cmd_status,
    }
    if args.comando is None:
        parser.print_help()
    else:
        cmds[args.comando](args)

if __name__ == "__main__":
    main()
