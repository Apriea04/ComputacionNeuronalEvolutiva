"""Crea una matriz de distancias entre ciudades"""
import numpy as np
import random

def crear_matriz(size: int, path: str):
    """Crea una matriz de distancias entre ciudades"""

    matriz = np.zeros((size, size))
    for i in range(size):
        for j in range(i + 1, size):
            matriz[i][j] = random.randint(1, 100)
            matriz[j][i] = matriz[i][j]
    np.savetxt(path, matriz, fmt="%d")
    return matriz

if __name__ == "__main__":
    crear_matriz(10000, "Viajante/Datos/10k.data")