"""
Preprocesador de texto para NLP en español.

Pipeline:
  1. Lowercase
  2. Eliminar caracteres especiales y números
  3. Tokenizar
  4. Eliminar stopwords en español (incrustadas, sin descarga)
  5. (Opcional) Stemming con SnowballStemmer

El resultado es una cadena de texto limpia lista para TF-IDF.
"""

import re
import sys
import unicodedata
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

try:
    from nltk.stem import SnowballStemmer
    stemmer = SnowballStemmer("spanish")
    STEMMER_AVAILABLE = True
except Exception:
    STEMMER_AVAILABLE = False

# ─── Stopwords en español (incrustadas) ──────────────────────────────────────
STOPWORDS_ES = {
    "de","la","que","el","en","y","a","los","del","se","las","por","un","para",
    "con","una","su","al","lo","como","mas","pero","sus","le","ya","o","este",
    "si","porque","esta","entre","cuando","muy","sin","sobre","tambien","me",
    "hasta","hay","donde","quien","desde","todo","nos","durante","todos","uno",
    "les","ni","contra","otros","ese","eso","ante","ellos","e","esto","mi",
    "antes","algunos","que","unos","yo","otro","otras","otra","el","tanto","esa",
    "estos","mucho","quienes","nada","muchos","cual","poco","ella","estar","estas",
    "algunas","algo","nosotros","mis","tu","te","ti","tus","ellas","nosotras",
    "os","vuestro","vuestra","vuestros","vuestras","esos","esas","estoy","estas",
    "esta","estamos","estan","estuve","estuvo","estuvimos","siendo","sido",
    "tengo","tienes","tiene","tenemos","tienen","he","has","hemos","han","fue",
    "era","sea","sean","son","ser","ha","asi","bien","tan","sino","aun","aunque",
    "cada","aqui","ahi","alli","dos","tres","ademas","tras","hacia","segun",
    # Específicas de documentos formales:
    "senor","senora","don","dona","ltd","ltda","spa","fono","correo","telefono",
    "presente","mediante","favor","atte","atentamente","saludos","cordialmente",
    "dicho","dicha","mismo","misma","respectivo","respectiva","referido","referida",
    "citado","citada","indicado","indicada","suscrito","suscrita","mencionado",
}


def limpiar_texto(texto: str, aplicar_stemming: bool = False) -> str:
    """
    Limpia y normaliza un texto en español.

    Parámetros
    ----------
    texto            : texto crudo
    aplicar_stemming : si True, aplica stemming (reduce palabras a raíz)

    Retorna
    -------
    str con el texto limpio y normalizado.
    """
    if not isinstance(texto, str) or not texto.strip():
        return ""

    # 1. Minúsculas
    texto = texto.lower()

    # 2. Saltos de línea → espacio
    texto = re.sub(r"[\n\r\t]", " ", texto)

    # 3. Eliminar RUTs chilenos (XX.XXX.XXX-X)
    texto = re.sub(r"\d{1,2}\.\d{3}\.\d{3}-[\dkK]", "", texto)

    # 4. Eliminar montos ($XX.XXX)
    texto = re.sub(r"\$[\d\.]+", "", texto)

    # 5. Eliminar fechas (DD/MM/YYYY)
    texto = re.sub(r"\d{1,2}/\d{1,2}/\d{4}", "", texto)

    # 6. Eliminar números solos
    texto = re.sub(r"\b\d+\b", "", texto)

    # 7. Eliminar acentos (normalizar a ASCII)
    nfd = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    texto = unicodedata.normalize("NFC", texto)

    # 8. Eliminar caracteres no alfabéticos
    texto = re.sub(r"[^a-z\s]", " ", texto)

    # 9. Tokenizar
    tokens = texto.split()

    # 10. Filtrar stopwords y tokens cortos
    tokens = [t for t in tokens if t not in STOPWORDS_ES and len(t) > 2]

    # 11. (Opcional) Stemming
    if aplicar_stemming and STEMMER_AVAILABLE:
        tokens = [stemmer.stem(t) for t in tokens]

    return " ".join(tokens)


def preprocesar_corpus(df: pd.DataFrame,
                       col_texto: str = "texto",
                       aplicar_stemming: bool = False) -> pd.DataFrame:
    """
    Aplica limpiar_texto() a toda la columna de texto del DataFrame.

    Retorna DataFrame con columna adicional 'texto_limpio'.
    """
    df = df.copy()
    df["texto_limpio"] = df[col_texto].apply(
        lambda t: limpiar_texto(t, aplicar_stemming=aplicar_stemming)
    )

    n_vacios = (df["texto_limpio"].str.strip() == "").sum()
    if n_vacios > 0:
        print(f"[WARN] {n_vacios} docs vacíos tras preprocesar → eliminados.")
        df = df[df["texto_limpio"].str.strip() != ""].reset_index(drop=True)

    return df


if __name__ == "__main__":
    from config import DATASET_CSV
    df = pd.read_csv(DATASET_CSV)
    df = preprocesar_corpus(df)
    row = df.iloc[0]
    print("ORIGINAL:", row["texto"][:200])
    print("LIMPIO:  ", row["texto_limpio"][:200])
    print(f"\nTokens promedio: {df['texto_limpio'].str.split().str.len().mean():.0f}")
