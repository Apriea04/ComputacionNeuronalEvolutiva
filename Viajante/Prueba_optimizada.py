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

def crear_poblacion_optimizada(
    num_poblaciones: int,
    tam_poblacion: int,
    aptitud: Callable,
    matriz_adyacencia: np.ndarray,
    verbose: bool = False,
) -> np.ndarray:
    """
    Crea una población de individuos utilizando NumPy para optimizar los cálculos y la generación de números aleatorios.
    
    :param num_poblaciones: Número de poblaciones.
    :param tam_poblacion: Tamaño de la población.
    :param aptitud: Función de aptitud.
    :param matriz_adyacencia: Array de NumPy que representa las distancias entre los nodos.
    :param verbose: Si es True, imprime información del progreso.
    :return: Array de NumPy que representa la población.
    """
    poblacion = np.zeros((tam_poblacion, num_poblaciones), dtype=int)

    for i in range(tam_poblacion):
        resto_recorrido = np.random.permutation(np.arange(1, num_poblaciones))
        individuo = np.concatenate(([0], resto_recorrido))

        while aptitud(individuo, matriz_adyacencia) == float("inf"):
            resto_recorrido = np.random.permutation(np.arange(1, num_poblaciones))
            individuo = np.concatenate(([0], resto_recorrido))

        poblacion[i] = individuo

        if verbose:
            print(
                "Creados {n} de {t} individuos".format(
                    n=i + 1, t=tam_poblacion
                ),
                end="\r",
            )

    if verbose:
        print()

    return poblacion

def mutar_optimizada(individuo: np.ndarray) -> np.ndarray:
    """
    Muta un individuo utilizando NumPy. Puede que el camino resultante no sea válido.
    
    :param individuo: Array de NumPy de enteros que representa el camino.
    :return: Array de NumPy de enteros que representa el camino mutado.
    """
    # Realizamos una copia del individuo para evitar modificar el original
    mutado = individuo.copy()
    # Elegimos dos posiciones aleatorias dentro del array
    pos1, pos2 = np.random.randint(0, len(mutado), size=2)
    # Intercambiamos los valores de las dos posiciones elegidas
    mutado[pos1], mutado[pos2] = mutado[pos2], mutado[pos1]
    return mutado

def crossover_partially_mapped_optimizado(lista_padres: np.ndarray, aptitud: Callable, matriz_adyacencia: np.ndarray, probabilidad: float) -> np.ndarray:
    """Realiza el crossover partially mapped de manera optimizada utilizando NumPy.
    
    :param lista_padres: Array de NumPy de todos los padres a cruzar.
    :param aptitud: Función de aptitud.
    :param matriz_adyacencia: Array de NumPy que representa las distancias entre los nodos.
    :param probabilidad: Probabilidad de que se realice el crossover.
    :return: Array de NumPy de hijos.
    """
    num_padres = lista_padres.shape[0]
    longitud_individuo = lista_padres.shape[1]
    lista_hijos = []

    for i in range(0, num_padres, 2):
        padre_1 = lista_padres[i]
        padre_2 = lista_padres[i + 1]

        if random.random() > probabilidad:
            lista_hijos.append(padre_1)
            lista_hijos.append(padre_2)
            continue

        punto_corte_1, punto_corte_2 = sorted(random.sample(range(longitud_individuo), 2))

        hijo_1 = np.full(longitud_individuo, -1)
        hijo_2 = np.full(longitud_individuo, -1)

        # Intercambiamos los segmentos entre los padres y los hijos
        hijo_1[punto_corte_1:punto_corte_2] = padre_2[punto_corte_1:punto_corte_2]
        hijo_2[punto_corte_1:punto_corte_2] = padre_1[punto_corte_1:punto_corte_2]

        # Función para completar el resto del hijo basado en las reglas del PMX
        def completar_hijo(hijo, padre_opuesto, segmento_opuesto):
            for idx, valor in enumerate(padre_opuesto):
                if idx >= punto_corte_1 and idx < punto_corte_2:
                    continue
                while valor not in hijo:
                    if valor in segmento_opuesto:
                        valor = padre_opuesto[np.where(segmento_opuesto == valor)[0][0]]
                    else:
                        hijo[np.where(hijo == -1)[0][0]] = valor
                        break

        completar_hijo(hijo_1, padre_1, hijo_2[punto_corte_1:punto_corte_2])
        completar_hijo(hijo_2, padre_2, hijo_1[punto_corte_1:punto_corte_2])

        # Añadimos a la lista de hijos si son válidos
        if aptitud(hijo_1, matriz_adyacencia) != float('inf'):
            lista_hijos.append(hijo_1)
        else:
            lista_hijos.append(padre_1)

        if aptitud(hijo_2, matriz_adyacencia) != float('inf'):
            lista_hijos.append(hijo_2)
        else:
            lista_hijos.append(padre_2)

    return np.array(lista_hijos)

