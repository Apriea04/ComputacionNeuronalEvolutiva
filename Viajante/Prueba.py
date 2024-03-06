import random
from typing import Callable


def leer_distancias(path: str) -> tuple:
    """Lee una matriz distancias de un fichero de texto.
    :param path: Ruta del fichero.
    :return: lista de nombres, matriz de distancias.
    """
    # Inicializamos una lista para los nombres de los municipios y una lista de listas para la matriz de números
    nombres_municipios = []
    matriz_numeros = []

    # Abrimos el archivo para lectura
    with open(path, "r", encoding="utf-8") as archivo:
        # Iteramos sobre cada línea del archivo
        for linea in archivo:
            # Dividimos la línea en dos partes: el nombre del municipio y los números
            # Primero eliminamos las comillas y luego dividimos usando el primer espacio encontrado
            partes = linea.strip().split('"')
            nombre_municipio = partes[1].strip()
            numeros = partes[2].strip().split()

            # Añadimos el nombre del municipio a la lista de nombres
            nombres_municipios.append(nombre_municipio)

            # Convertimos los números a enteros y los añadimos a la matriz
            matriz_numeros.append([int(numero) for numero in numeros])

    # Devolvemos los nombres y la matriz
    return nombres_municipios, matriz_numeros


def aptitud_viajante(individuo: list, matriz_adyacencia) -> float:
    """Devuelve la aptitud de un individuo. Se define como la suma de costes (distancias) de recorrer el camino que indica el individuo.
    Elementos a tener en cuenta para el cálculo de la aptitud:
    - El viajante tiene ubicación de salida fija, el final puede ser cualquier población.
    :param individuo: Lista de enteros que representa el camino.
    :param matriz_adyacencia: Matriz de adyacencia que representa las distancias entre los nodos. Los valores de las coordenadas (i,i) corresponden al coste de estar en la población correspondiente.
    :return: Valor de aptitud del individuo.
    """
    # Ayuda 1: Si no comenzamos en el almacén (población 0), la aptitud será infinita
    if individuo[0] != 0:
        return float("inf")

    # Aptitud comienza como el tiempo en almacén
    aptitud = matriz_adyacencia[individuo[0]][individuo[0]]

    for ciudad in range(1, len(individuo)):
        # Sumamos el coste de para llegar a la población actual desde la anterior
        aptitud += matriz_adyacencia[individuo[ciudad]][individuo[ciudad + 1]]

        # Sumamos el coste de estar en la población actual
        aptitud += matriz_adyacencia[individuo[ciudad]][individuo[ciudad]]

    return aptitud


def crear_poblacion(num_poblaciones: int, tam_poblacion: int) -> list:
    """Crea una población de individuos.
    :param num_poblaciones: Número de poblaciones.
    :param tam_poblacion: Tamaño de la población.
    :return: Lista de listas que representan individuos.
    """
    # Creamos una lista de listas con el tamaño de la población
    poblacion = []
    for i in range(tam_poblacion):
        # Creamos un individuo aleatorio
        individuo = list(range(num_poblaciones))
        random.shuffle(individuo)
        print(individuo)
        poblacion.append(individuo)
    return poblacion


def mutar(individuo: list) -> list:
    """Muta un individuo.
    :param individuo: Lista de enteros que representa el camino.
    :return: Lista de enteros que representa el camino mutado.
    """
    # Copiamos el individuo
    mutado = individuo[:]
    # Elegimos dos posiciones aleatorias
    pos1 = random.randint(0, len(mutado) - 1)
    pos2 = random.randint(0, len(mutado) - 1)
    # Intercambiamos los valores de las posiciones
    mutado[pos1], mutado[pos2] = mutado[pos2], mutado[pos1]
    return mutado


# https://www.hindawi.com/journals/cin/2017/7430125/


def crossover_partially_mapped(lista_padres: list, aptitud: Callable) -> list:
    """Realiza el crossover partially mapped según se explica en https://www.hindawi.com/journals/cin/2017/7430125/
    Resumidamente, se van eligiendo padres en orden y de 2 en 2.
    Cada 2 padres producen 2 hijos.
    :param lista_padres: Lista de todos los padres a cruzar.
    :param aptitud: Función de aptitud.
    :return: Lista de hijos.
    """
    # Incializamos la lista de hijos
    lista_hijos = []

    # Iteramos sobre los padres de 2 en 2
    for i in range(0, len(lista_padres), 2):
        # Nombramos los padres para facilidad de uso
        padre_1 = lista_padres[i]
        padre_2 = lista_padres[i + 1]

        # Elegimos dos puntos de corte aleatorios
        punto_corte_1, punto_corte_2 = sorted(random.sample(range(len(padre_1)), 2))

        # Inicializamos los hijos
        hijo_1 = [-1 for _ in range(len(padre_1))]
        hijo_2 = [-1 for _ in range(len(padre_1))]

        # El intervalo entre los puntos de corte es intercambiado e insertado en la misma posición en los hijos
        hijo_2[punto_corte_1:punto_corte_2] = padre_1[punto_corte_1:punto_corte_2]
        hijo_1[punto_corte_1:punto_corte_2] = padre_2[punto_corte_1:punto_corte_2]

        # Pasamos los número originales de los padres a los hijos siempre y cuando no estén ya presentes en ellos:
        # Aprovechamos a crear una lista para los números que ya están en los hijos
        ya_en_hijo_1 = []
        ya_en_hijo_2 = []

        for j in range(len(padre_1)):
            # Vamos recorriendo los padres por fuera de la zona traspuesta
            # TODO: este if primero probablemente no sea necesario, ya que la comprobación es redundante
            if j < punto_corte_1 or j >= punto_corte_2:
                if padre_2[j] not in hijo_2:
                    hijo_2[j] = padre_2[j]
                else:
                    ya_en_hijo_2.append(padre_2[j])
                if padre_1[j] not in hijo_1:
                    hijo_1[j] = padre_1[j]
                else:
                    ya_en_hijo_1.append(padre_1[j])

        # Completamos los hijos con los números que faltan EN ORDEN
        for j in range(len(hijo_1)):
            if hijo_1[j] == -1:
                hijo_1[j] = ya_en_hijo_1.pop(0)
            if hijo_2[j] == -1:
                hijo_2[j] = ya_en_hijo_2.pop(0)

        # Añadimos los hijos a la lista de hijos
        lista_hijos.append(hijo_1)
        lista_hijos.append(hijo_2)

    return lista_hijos


def crossover_order(lista_padres: list, aptitud: Callable) -> list:
    """Realiza el order crossover según se explica en https://www.hindawi.com/journals/cin/2017/7430125/
    Resumidamente, se van eligiendo padres en orden y de 2 en 2.
    Cada 2 padres producen 2 hijos.
    :param lista_padres: Lista de todos los padres a cruzar.
    :param aptitud: Función de aptitud.
    :return: Lista de hijos.
    """
    # Incializamos la lista de hijos
    lista_hijos = []

    # Iteramos sobre los padres de 2 en 2
    for i in range(0, len(lista_padres), 2):
        # Nombramos los padres para facilidad de uso
        padre_1 = lista_padres[i]
        padre_2 = lista_padres[i + 1]

        # Elegimos dos puntos de corte aleatorios
        punto_corte_1, punto_corte_2 = sorted(random.sample(range(len(padre_1)), 2))

        # Inicializamos los hijos
        hijo_1 = [-1 for _ in range(len(padre_1))]
        hijo_2 = [-1 for _ in range(len(padre_1))]

        # El intervalo entre los puntos de corte es pasado directamente a los hijos
        hijo_2[punto_corte_1:punto_corte_2] = padre_2[punto_corte_1:punto_corte_2]
        hijo_1[punto_corte_1:punto_corte_2] = padre_1[punto_corte_1:punto_corte_2]

        # Obtenemos la lista de los números de cada padre a partir del punto final de corte ordenados por aparición
        lista_padre_1 = [padre_1[i] for i in range(punto_corte_2, len(padre_1))] + [
            padre_1[i] for i in range(0, punto_corte_2)
        ]
        lista_padre_2 = [padre_2[i] for i in range(punto_corte_2, len(padre_2))] + [
            padre_2[i] for i in range(0, punto_corte_2)
        ]

        # Eliminamos los números que ya están en el hijo contrario
        # TODO: intentar fusionar estos dos bucles en uno solo
        for numero in lista_padre_1:
            if numero in hijo_2:
                lista_padre_1.remove(numero)
        for numero in lista_padre_2:
            if numero in hijo_1:
                lista_padre_2.remove(numero)

        # Completamos los hijos con los números que faltan
        for j in range(len(hijo_1)):
            if hijo_1[j] == -1:
                hijo_1[j] = lista_padre_2.pop(0)
            if hijo_2[j] == -1:
                hijo_2[j] = lista_padre_1.pop(0)

        # Añadimos los hijos a la lista de hijos
        lista_hijos.append(hijo_1)
        lista_hijos.append(hijo_2)

    return lista_hijos
