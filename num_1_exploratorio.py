import matplotlib.pyplot as plt
import pandas as pd

from utilities import save_figure, summary_table

# 1. Realizar análisis descriptivo y exploratorio exhaustivo que incluya el
# cálculo de medidas de tendencia central y variabilidad, la generación de
# gráficos descriptivos y el cruce entre variables. Este análisis debe permitir
# identificar patrones en los datos y extraer conclusiones clave.


# analisis descriptivo y exploratorio
def num_1_exploratorio(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("1. Análisis descriptivo y exploratorio")
    print("=" * 70)

    print("\n-- Dimensiones del conjunto de datos --")
    print(f"Miembros de filas: {df.shape[0]}, Variables: {df.shape[1]}")

    # medidas de tendencia central y variabilidad
    summary = summary_table(df)
    print(summary.to_string())

    # verificamos los duplicados
    duplicated = df.duplicated().sum()
    print(f"\nFilas duplicadas: {duplicated}")

    # Histogramas
    numerics = df.select_dtypes(include="number").columns.tolist()
    fig, ejes = plt.subplots(4, 3, figsize=(14, 12), layout="constrained")
    ejes = ejes.ravel()
    for i, col in enumerate(numerics):
        ejes[i].hist(df[col], bins=25, color="#4c72b0", edgecolor="white")
        ejes[i].set_title(f"Distribución de {col}", fontsize=10)
        ejes[i].set_xlabel(col)
        ejes[i].set_ylabel("Frecuencia")
    fig.suptitle("Histogramas de las variables cuantitativas", fontsize=13, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(fig, "01_histogramas.png")

    # Diagramas de caja
    fig, ejes = plt.subplots(4, 3, figsize=(14, 12), layout="constrained")
    ejes = ejes.ravel()
    for i, col in enumerate(numerics):
        ejes[i].boxplot(df[col], vert=False)
        ejes[i].set_title(f"Diagrama de caja de {col}", fontsize=10)
        ejes[i].set_xlabel(col)
    fig.suptitle("Diagramas de caja de las variables cuantitativas", fontsize=13)
    fig.tight_layout()
    save_figure(fig, "02_diagramas_caja.png")

    # Cruce de variables
    print("\nCruce IMC por genero\n")
    print(
        df.groupby("Genero")["IMC"].agg(["count", "mean", "std", "min", "max"]).round(2)
    )

    print("\nCruce Calorias quemadas por tipo de entrenamiento\n")
    print(
        df.groupby("Tipo_entrenamiento")["Calorias_quemadas"]
        .agg(["count", "mean", "std", "min", "max"])
        .round(2)
    )

    print("\n Cruce IMC por nivel de experiencia")
    print(df.groupby("Nivel_experiencia")["IMC"].agg(["count", "mean", "std"]).round(2))

    print("\n Distribución de la categoría de IMC\n")
    print(df["Categoria_IMC"].value_counts(dropna=False).to_string())

    # Gráfico de IMC por categoria
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    df["Categoria_IMC"].value_counts(dropna=False).plot(
        kind="bar", ax=ejes[0], color="#4c72b0", edgecolor="white"
    )
    ejes[0].set_title("Frecuencia de miembros por categoria de IMC")
    ejes[0].set_xlabel("Categoria de IMC")
    ejes[0].set_ylabel("Número de miembros")

    tipos = df["Tipo_entrenamiento"].unique()
    datos = [df[df["Tipo_entrenamiento"] == t]["Calorias_quemadas"] for t in tipos]
    ejes[1].boxplot(datos, tick_labels=tipos)
    ejes[1].set_title("Calorias quemadas por tipo de entrenamiento")
    ejes[1].set_ylabel("Calorias quemadas")
    fig.tight_layout()
    save_figure(fig, "03_cruces_variables.png")
