import random

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


def aptitud(individuo: list, matriz_adyacencia) -> float:
    """Devuelve la aptitud de un individuo. Se define como la suma de costes (distancias) de recorrer el camino que indica el individuo.
    Elementos a tener en cuenta para el cálculo de la aptitud:
    - El viajante tiene ubicación de salida fija, el final puede ser cualquier población.
    :param individuo: Lista de enteros que representa el camino.
    :param matriz_adyacencia: Matriz de adyacencia que representa las distancias entre los nodos. Los valores de las coordenadas (i,i) corresponden al coste de estar en la población correspondiente.
    :return: Valor de aptitud del individuo.
    """
    #Ayuda 1: Si no comenzamos en el almacén (población 0), la aptitud será infinita
    if individuo[0] != 0:
        return float('inf')
    
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