import random
from typing import Callable
import matplotlib.pyplot as plt
import numpy as np

# Todo este código está pensado desde el principio para minimizar la función de aptitud

NUM_ITERACIONES = 4000
PROB_MUTACION = 0.1
PROB_CRUZAMIENTO = 0.7
PARTICIPANTES_TORNEO = 3
NUM_INDIVIDUOS = 100


def leer_distancias_optimizada(path_distancias: str, path_nombres: str = None) -> tuple:
    """
    Lee una matriz de distancias de un fichero de texto utilizando NumPy para optimizar el proceso.
    Si se proporciona un path_nombres, lee los nombres desde ese archivo; de lo contrario, asume que
    están en el mismo archivo que las distancias, en líneas precedidas por nombres entrecomillados.
    
    :param path_distancias: Ruta del fichero de distancias.
    :param path_nombres: Ruta al fichero que contiene los nombres (opcional).
    :return: Tuple que contiene una lista de nombres y un array de distancias de NumPy.
    """
    if path_nombres is None:
        # Primero leemos todo el archivo para separar nombres de distancias
        with open(path_distancias, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        nombres_municipios = []
        distancias_str = []  # Guardaremos las líneas de distancias como strings para luego convertirlas juntas

        for line in lines:
            if '"' in line:  # Detectamos la presencia de un nombre
                nombre_municipio = line.split('"')[1].strip()
                nombres_municipios.append(nombre_municipio)
                distancias_str.append(line.split('"')[-1].strip())
            else:
                distancias_str.append(line.strip())

        # Usamos np.loadtxt con una StringIO para convertir las distancias a un array de NumPy
        from io import StringIO
        distancias_io = StringIO("\n".join(distancias_str))
        distancias = np.loadtxt(distancias_io)

    else:
        nombres_municipios = np.loadtxt(path_nombres, dtype=str, delimiter="\n").tolist()
        distancias = np.loadtxt(path_distancias, dtype=float)

    return nombres_municipios, distancias

def aptitud_viajante_optimizada(
    individuo: np.ndarray, matriz_adyacencia: np.ndarray, tiempo_total: bool = False
) -> float:
    """
    Devuelve la aptitud de un individuo utilizando NumPy para optimizar los cálculos.
    La aptitud se define como la suma de costes (distancias) de recorrer el camino indicado por el individuo.
    
    :param individuo: Array de NumPy de enteros que representa el camino.
    :param matriz_adyacencia: Array de NumPy que representa las distancias entre los nodos.
    :param tiempo_total: Si es True, se suma también el tiempo total en cada ciudad al coste.
    :return: Valor de aptitud del individuo.
    """
    # Verificar si el inicio es correcto
    if individuo[0] != 0:
        return float("inf")
    
    # Calcular las distancias entre nodos consecutivos
    distancias = matriz_adyacencia[individuo[:-1], individuo[1:]]
    
    # Si hay algún 0 en las distancias, el camino no es válido
    if np.any(distancias == 0):
        return float("inf")
    
    aptitud = np.sum(distancias)
    
    if tiempo_total:
        # Añadir el tiempo de estancia en cada ciudad, incluyendo la inicial y la final
        tiempo_estancia = matriz_adyacencia[individuo, individuo]
        aptitud += np.sum(tiempo_estancia)
    
    return aptitud
