import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from utilities import save_figure

SEED = 42


def numeral_4_kmeans(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("NUMERAL 4: AGRUPACION CON K-MEANS")
    print("=" * 70)

    variables_cluster = ["Calorias_quemadas", "Porcentaje_grasa", "IMC"]
    X = df[variables_cluster].to_numpy()

    # Estandarizacion necesaria: las tres variables tienen escalas distintas
    # (cientos de calorias frente a porcentajes de dos digitos) y K-means
    # minimiza distancias euclidianas, por lo que sin escalar dominaria la
    # variable de mayor rango.
    escalador = StandardScaler()
    X_std = escalador.fit_transform(X)

    # Grafico de codo: inercia (suma de distancias cuadraticas al centroide)
    # para k = 1..10. El "codo" indica el k a partir del cual agregar grupos
    # deja de aportar una reduccion relevante de la inercia.
    ks = range(1, 11)
    inercias = [KMeans(n_clusters=k, random_state=SEED, n_init=10)
                .fit(X_std).inertia_ for k in ks]

    # Puntaje de silueta como apoyo objetivo para elegir k (solo k >= 2)
    siluetas = {k: silhouette_score(X_std, KMeans(
        n_clusters=k, random_state=SEED, n_init=10).fit_predict(X_std))
        for k in range(2, 7)}

    print("Inercia por numero de grupos (grafico de codo):")
    print(pd.DataFrame({"inercia": np.round(inercias, 1)}, index=list(ks)).to_string())
    print("\nPuntaje de silueta medio por numero de grupos:")
    print(pd.Series(siluetas).round(3).to_string())

    k_ideal = max(siluetas, key=siluetas.get)
    print(f"\nNumero de grupos ideal: k = {k_ideal} "
          f"(maximo puntaje de silueta = {siluetas[k_ideal]:.3f}, "
          f"coincidente con el codo del grafico).")

    # Modelo final con el k elegido
    modelo = KMeans(n_clusters=k_ideal, random_state=SEED, n_init=10)
    etiquetas = modelo.fit_predict(X_std)
    df = df.copy()
    df["cluster"] = etiquetas

    # Centroides en las unidades originales (se revierte la estandarizacion)
    centroides = escalador.inverse_transform(modelo.cluster_centers_)
    print("\nCentroides de los grupos en unidades originales:")
    print(pd.DataFrame(centroides.round(2), columns=variables_cluster,
                       index=[f"Grupo {i}" for i in range(k_ideal)]).to_string())

    print("\nTamano de cada grupo:")
    print(df["cluster"].value_counts().sort_index().to_string())

    print("\nInterpretacion de los grupos (promedios por cluster):")
    print(df.groupby("cluster")[variables_cluster].mean().round(2).to_string())

    # Graficos
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.5))
    ejes[0].plot(ks, inercias, marker="o", color="#4c72b0")
    ejes[0].axvline(k_ideal, color="red", linestyle="--",
                    label=f"k elegido = {k_ideal}")
    ejes[0].set_xlabel("Numero de grupos (k)")
    ejes[0].set_ylabel("Inercia (WCSS)")
    ejes[0].set_title("Grafico de codo")
    ejes[0].legend(fontsize=8)

    ejes[1].plot(list(siluetas.keys()), list(siluetas.values()),
                 marker="s", color="#dd8452")
    ejes[1].axvline(k_ideal, color="red", linestyle="--",
                    label=f"k elegido = {k_ideal}")
    ejes[1].set_xlabel("Numero de grupos (k)")
    ejes[1].set_ylabel("Puntaje de silueta medio")
    ejes[1].set_title("Silueta media por numero de grupos")
    ejes[1].legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, "07_codo_silueta.png")

    colores = plt.get_cmap("viridis")(np.linspace(0, 1, k_ideal))
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.5))
    for k in range(k_ideal):
        sub = df[df["cluster"] == k]
        ejes[0].scatter(sub["Calorias_quemadas"], sub["IMC"],
                        s=15, alpha=0.6, color=colores[k], label=f"Grupo {k}")
        ejes[1].scatter(sub["Porcentaje_grasa"], sub["IMC"],
                        s=15, alpha=0.6, color=colores[k])
    for ax, (xvar, yvar) in zip(ejes, [("Calorias_quemadas", "IMC"),
                                       ("Porcentaje_grasa", "IMC")]):
        ax.set_xlabel(xvar)
        ax.set_ylabel(yvar)
    ejes[0].set_title("Grupos K-means: calorias quemadas vs IMC")
    ejes[1].set_title("Grupos K-means: porcentaje de grasa vs IMC")
    ejes[0].legend(fontsize=8, title="Cluster")
    fig.tight_layout()
    save_figure(fig, "08_clusters.png")

    return df


# Alias para mantener consistencia de nombres con los demas modulos
num_4_kmeans = numeral_4_kmeans
