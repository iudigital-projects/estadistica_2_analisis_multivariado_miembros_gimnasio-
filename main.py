import os
import sys

from num_1_exploratorio import num_1_exploratorio
from num_2_correlacion import num_2_correlacion
from num_3_pca import num_3_pca
from num_4_kmeans import num_4_kmeans
from utilities import DATA_PATH, FIGURES_PATH, prepare_data


def run_all(df) -> None:
    num_1_exploratorio(df)
    num_2_correlacion(df)
    num_3_pca(df)
    num_4_kmeans(df)


def menu() -> None:
    os.makedirs(FIGURES_PATH, exist_ok=True)
    df = prepare_data(DATA_PATH)

    opciones = {
        "1": ("Analisis descriptivo y exploratorio", lambda: num_1_exploratorio(df)),
        "2": ("Matriz de correlacion", lambda: num_2_correlacion(df)),
        "3": ("Analisis de componentes principales (PCA)", lambda: num_3_pca(df)),
        "4": ("Agrupacion con K-Means", lambda: num_4_kmeans(df)),
        "5": ("Ejecutar todos los numerales", lambda: run_all(df)),
    }

    while True:
        print("\n" + "=" * 50)
        print("MENU - ACTIVIDAD 3 (ESTADISTICA 2)")
        print("=" * 50)
        for key, (desc, _) in opciones.items():
            print(f"{key}. {desc}")
        print("0. Salir")
        opcion = input("Seleccione una opcion: ").strip()

        if opcion == "0":
            print("Saliendo...")
            break
        if opcion in opciones:
            print(f"\nEjecutando: {opciones[opcion][0]}")
            opciones[opcion][1]()
        else:
            print("Opcion invalida. Intente de nuevo.")


def main() -> None:
    # Modo no interactivo: python main.py <opcion>
    if len(sys.argv) > 1:
        os.makedirs(FIGURES_PATH, exist_ok=True)
        df = prepare_data(DATA_PATH)
        arg = sys.argv[1].lower()
        dispatch = {
            "1": num_1_exploratorio,
            "2": num_2_correlacion,
            "3": num_3_pca,
            "4": num_4_kmeans,
            "all": run_all,
            "todos": run_all,
        }
        if arg in dispatch:
            dispatch[arg](df)
            return
        print(f"Opcion desconocida: {arg}")
        return
    menu()


if __name__ == "__main__":
    main()