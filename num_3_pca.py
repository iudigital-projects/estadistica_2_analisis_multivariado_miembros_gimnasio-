import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utilities import save_figure

SEED=42

# 3. Realice la reducción del número de variables que apliquen conservando el
# máximo de información posible. Este ejercicio hágalo mediante el algoritmo de
# análisis de componentes principales. Considera usted que para el conjunto de
# variables presentado es una buena decisión realizar una reducción de
# variables, las componentes encontradas logran hacer una buena representación de
# las variables originales

def num_3_pca(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("NUMERAL 3: ANALISIS DE COMPONENTES PRINCIPALES (ACP)")
    print("=" * 70)

    # Variables consideradas: todas las cuantitativas primarias.
    # Se excluye IMC porque es una combinacion determinista de Peso y Altura
    # (IMC = peso / altura^2); incluirlo duplicaria la informacion de esas dos
    # variables y distorsionaria el analisis.
    variables = [c for c in df.select_dtypes(include="number").columns
                 if c != "IMC"]
    print(f"Variables usadas en el ACP ({len(variables)}): {variables}")

    X = df[variables].to_numpy()

    # Estandarizacion: las variables tienen unidades distintas (anios, kg, m,
    # BPM, kcal, litros). Sin estandarizar, las de mayor varianza dominarian
    # las componentes. Se usa z = (x - media) / desviacion estandar.
    escalador = StandardScaler()
    X_std = escalador.fit_transform(X)

    acp = PCA(random_state=SEED)
    componentes = acp.fit_transform(X_std)

    varianza = acp.explained_variance_
    proporcion = acp.explained_variance_ratio_
    acumulada = np.cumsum(proporcion)

    print("\nVarianza explicada por componente:")
    print(pd.DataFrame({
        "valor_propio (varianza)": varianza.round(3),
        "proporcion_%": (proporcion * 100).round(2),
        "acumulada_%": (acumulada * 100).round(2),
    }, index=[f"CP{i+1}" for i in range(len(varianza))]).to_string())

    n_kaiser = int((varianza > 1).sum())
    n_80 = int(np.argmax(acumulada >= 0.80) + 1)
    print(f"\nComponentes con valor propio > 1 (criterio de Kaiser): {n_kaiser}")
    print(f"Componentes necesarias para explicar al menos el 80 % de la "
          f"varianza: {n_80}")

    # Cargas factoriales: correlacion entre cada variable original y cada
    # componente. Se usan para interpretar que representa cada componente.
    cargas = acp.components_.T * np.sqrt(varianza)
    tabla_cargas = pd.DataFrame(cargas[:, :3].round(2), index=variables,
                                columns=["CP1", "CP2", "CP3"])
    print("\nCargas de las primeras 3 componentes (correlaciones):")
    print(tabla_cargas.to_string())

    # ---------------- Graficos del ACP ----------------
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.5))
    cp = range(1, len(varianza) + 1)
    ejes[0].bar(cp, varianza, color="#4c72b0", edgecolor="white")
    ejes[0].axhline(1, color="red", linestyle="--", label="Criterio de Kaiser (valor propio = 1)")
    ejes[0].set_xlabel("Componente principal")
    ejes[0].set_ylabel("Valor propio (varianza explicada)")
    ejes[0].set_title("Grafico de sedimentacion (scree plot)")
    ejes[0].legend(fontsize=8)

    ejes[1].plot(cp, acumulada * 100, marker="o", color="#4c72b0")
    ejes[1].axhline(80, color="red", linestyle="--", label="Umbral del 80 %")
    ejes[1].set_xlabel("Numero de componentes")
    ejes[1].set_ylabel("Varianza explicada acumulada (%)")
    ejes[1].set_title("Varianza explicada acumulada")
    ejes[1].legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, "05_pca_varianza.png")

    fig, ax = plt.subplots(figsize=(7.5, 6))
    dispersion = ax.scatter(componentes[:, 0], componentes[:, 1],
                            c=df["IMC"], cmap="viridis", s=18, alpha=0.7)
    fig.colorbar(dispersion, ax=ax, label="IMC")
    ax.set_xlabel(f"CP1 ({proporcion[0]*100:.1f} % de la varianza)")
    ax.set_ylabel(f"CP2 ({proporcion[1]*100:.1f} % de la varianza)")
    ax.set_title("Proyeccion de los miembros en el plano CP1-CP2")
    fig.tight_layout()
    save_figure(fig, "06_pca_proyeccion.png")

