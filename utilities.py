import os
import unicodedata

import matplotlib.pyplot as plt
import pandas as pd

FIGURES_PATH = "figures"
DATA_PATH = "Miembros_gimnasio.xlsx"


def delete_accents(texto: str) -> str:
    """Eliminar acentos de una cadena de texto."""
    form = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in form if not unicodedata.combining(c))


def prepare_data(path: str) -> pd.DataFrame:
    """
    Ya que la columna viene con el nombre "Reposo_BPMBPM"
    la renombramos a "Reposo_BPM". Adicionalmente
    normalizamos los nombres de las columnas quitando
    las tildes.
    """
    df = pd.read_excel(path)
    df = df.rename(columns={"Reposo_BPMBPM": "Reposo_BPM"})
    df.columns = [delete_accents(c.strip()) for c in df.columns]
    return df


def figure_exists(name: str) -> bool:
    """Indica si la figura ya fue generada previamente."""
    return os.path.exists(os.path.join(FIGURES_PATH, name))


def save_figure(fig, name: str) -> str:
    """
    Guardamos las imagenes en figures/. Si el archivo ya existe, no se
    vuelve a generar/sobrescribir (guardia).
    """
    path = os.path.join(FIGURES_PATH, name)
    os.makedirs(FIGURES_PATH, exist_ok=True)
    if os.path.exists(path):
        print(f"  Figura ya existe, se omite: {path}")
        plt.close(fig)
        return path
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f" Figured saved: {path}")
    return path


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sacamos las medidas de tendencia central y variabilidad de las variables numericas

    mediana (50%), moda, varianza, coeficiente de variacion, asimetria y curtosis
    """
    numerics = df.select_dtypes(include="number")
    summary = numerics.agg(["mean", "median", "std", "var", "min", "max"])

    summary.loc["moda"] = numerics.mode().iloc[0]
    summary.loc["rango"] = summary.loc["max"] - summary.loc["min"]
    summary.loc["cv_%"] = summary.loc["std"] / summary.loc["mean"] * 100
    summary.loc["asimetria"] = numerics.skew()
    summary.loc["curtosis"] = numerics.kurtosis()
    return summary.round(2)
