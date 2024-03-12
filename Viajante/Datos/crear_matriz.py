"""Crea una matriz de distancias entre ciudades"""
import numpy as np
import random

def crear_matriz_distancias(size: int, path: str):
    """Crea una matriz de distancias entre ciudades"""

    matriz = np.zeros((size, size))
    for i in range(size):
        for j in range(i + 1, size):
            matriz[i][j] = random.randint(1, 100)
            matriz[j][i] = matriz[i][j]
    np.savetxt(path, matriz, fmt="%d")
    return matriz

def crear_lista_coordenadas(size: int, path: str, min: float = -100, max: float = 100):
    """Crea una lista de coordenadas de ciudades en formato (x, y) con valores entre min y max"""
    coordenadas = []
    for i in range(size):
        x = random.uniform(min, max)
        y = random.uniform(min, max)
        coordenadas.append((x, y))
    with open(path, "w") as file:
        for i in range(size):
            file.write(f"{coordenadas[i][0]} {coordenadas[i][1]}\n")
    return coordenadas

def coordenadas_a_distancias(path_coordenadas: str, path_distancias: str, distancia: str = "euclidea"):
    """Lee el fichero con las coordenadas y produce  la matriz de distancias correspondiente"""
    
    # Leer las coordenadas
    with open(path_coordenadas, "r") as file:
        coordenadas = file.readlines()
    coordenadas = [tuple(map(float, coord.split())) for coord in coordenadas]
    size = len(coordenadas)
    
    matriz = np.zeros((size, size))
    
    for i in range(size):
        for j in range(i + 1, size):
            match distancia:
                case "euclidea":
                    matriz[i][j] = np.sqrt((coordenadas[i][0] - coordenadas[j][0])**2 + (coordenadas[i][1] - coordenadas[j][1])**2)
                case "manhattan":
                    matriz[i][j] = abs(coordenadas[i][0] - coordenadas[j][0]) + abs(coordenadas[i][1] - coordenadas[j][1])
                case "chebyshev":
                    matriz[i][j] = max(abs(coordenadas[i][0] - coordenadas[j][0]), abs(coordenadas[i][1] - coordenadas[j][1]))
            matriz[j][i] = matriz[i][j]
    np.savetxt(path_distancias, matriz, fmt="%d")
    

if __name__ == "__main__":
    crear_lista_coordenadas(10, "Viajante/Datos/10_coordenadas.txt")
    coordenadas_a_distancias("Viajante/Datos/10_coordenadas.txt", "Viajante/Datos/10_distancias.txt")
        