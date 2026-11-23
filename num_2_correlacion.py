import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utilities import save_figure

# Realice la matriz de correlación con las variables que apliquen y haga la
# representación gráfica de la misma para las variables presentadas, concluya a
# partir de los resultados encontrados.

# Matriz de correlacion
def num_2_correlacion(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("2. MATRIZ DE CORRELACION")
    print("=" * 70)

    numerics = df.select_dtypes(include="number")
    corr = numerics.corr(method="pearson").round(2)
    print(corr.to_string())

    # Mapa de calor con matplotlib
    fig, ax = plt.subplots(figsize=(10, 8))
    imagen = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, corr.iloc[i, j], ha="center", va="center",
                    fontsize=7, color="black")
    fig.colorbar(imagen, ax=ax, label="Coeficiente de Pearson")
    ax.set_title("Matriz de correlacion (Pearson) de las variables cuantitativas")
    fig.tight_layout()
    print("\n")
    save_figure(fig, "04_matriz_correlacion.png")

    # Los pares con correlacion mas fuerte en valor absoluto
    pares = (corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
             .stack().sort_values(key=abs, ascending=False))
    print("\nPares de variables con correlacion mas fuerte (valor absoluto): \n")
    print(pares.head(8).to_string())
    return corr

