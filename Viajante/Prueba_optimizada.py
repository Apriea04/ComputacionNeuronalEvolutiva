import random
from typing import Callable
import matplotlib.pyplot as plt
import numpy as np

# Todo este código está pensado desde el principio para minimizar la función de aptitud

NUM_ITERACIONES = 4000
PROB_MUTACION = 0.1
PROB_CRUZAMIENTO = 0.7
PARTICIPANTES_TORNEO = 3
NUM_INDIVIDUOS = 1000


def eliminar_ceros_matriz(matriz: np.ndarray) -> np.ndarray:
    """
    Elimina los ceros de una matriz de NumPy, sustituyéndolos por float("inf"), a excepción de la diagonal principal.

    :param matriz: Array de NumPy.
    :return: Array de NumPy sin ceros.
    """
    # Obtener dimensiones de la matriz
    filas, columnas = matriz.shape

    # Crear una copia de la matriz para no modificar la original
    matriz_sin_ceros = matriz.copy()

    # Iterar sobre cada elemento de la matriz
    for i in range(filas):
        for j in range(columnas):
            # Verificar si el elemento no está en la diagonal principal y es igual a cero
            if i != j and matriz_sin_ceros[i, j] == 0:
                # Sustituir el cero por infinito
                matriz_sin_ceros[i, j] = float("inf")

    return matriz_sin_ceros


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
        with open(path_distancias, "r", encoding="utf-8") as file:
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
        with open(path_nombres, "r", encoding="utf-8") as file:
            nombres = file.readlines()

        nombres = [nombre.strip() for nombre in nombres]
        nombres_municipios = np.array(nombres)

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
    eliminar_inaptos: bool = True,
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

        while eliminar_inaptos and aptitud(individuo, matriz_adyacencia) == float("inf"):
            resto_recorrido = np.random.permutation(np.arange(1, num_poblaciones))
            individuo = np.concatenate(([0], resto_recorrido))

        poblacion[i] = individuo

        if verbose:
            print(
                "Creados {n} de {t} individuos".format(n=i + 1, t=tam_poblacion),
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


def crossover_partially_mapped_optimizado(
    lista_padres: np.ndarray,
    aptitud: Callable,
    matriz_adyacencia: np.ndarray,
    probabilidad: float,
) -> np.ndarray:
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
        
        #Nombra los padres para facilidad de uso. El 0 se va a mantener siempre
        padre_1 = lista_padres[i][1:]
        padre_2 = lista_padres[i + 1][1:]

        if random.random() > probabilidad:
            lista_hijos.append(padre_1)
            lista_hijos.append(padre_2)
            continue

        punto_corte_1, punto_corte_2 = sorted(
            random.sample(range(longitud_individuo), 2)
        )

        hijo_1 = np.full(longitud_individuo-1, -1)
        hijo_2 = np.full(longitud_individuo-1, -1)

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
        
        #Juntamos el 0 inicial con el resto del genes
        hijo_1 = np.insert(hijo_1, 0, 0)
        hijo_2 = np.insert(hijo_2, 0, 0)

        # Añadimos a la lista de hijos si son válidos
        if aptitud(hijo_1, matriz_adyacencia) != float("inf"):
            lista_hijos.append(hijo_1)
        else:
            lista_hijos.append(np.insert(padre_1, 0, 0))

        if aptitud(hijo_2, matriz_adyacencia) != float("inf"):
            lista_hijos.append(hijo_2)
        else:
            lista_hijos.append(np.insert(padre_2, 0, 0))

    return np.array(lista_hijos)


def crossover_order_optimizado(
    lista_padres: np.ndarray,
    aptitud: Callable,
    matriz_adyacencia: np.ndarray,
    probabilidad: float,
) -> np.ndarray:
    """
    Realiza el order crossover según se explica en https://www.hindawi.com/journals/cin/2017/7430125/
    Resumidamente, se eligen padres de 2 en 2 y cada par produce 2 hijos.

    :param lista_padres: Array de NumPy de todos los padres a cruzar.
    :param aptitud: Función de aptitud.
    :param matriz_adyacencia: Array de NumPy que representa las distancias entre los nodos.
    :param probabilidad: Probabilidad de que se realice el crossover.
    :return: Array de NumPy que representa la lista de hijos.
    """
    num_padres, tam_individuo = lista_padres.shape

    # Inicializamos la matriz de hijos
    lista_hijos = np.zeros_like(lista_padres)

    # Iteramos sobre los padres de 2 en 2
    for i in range(0, num_padres, 2):
        if np.random.rand() > probabilidad:
            lista_hijos[i] = lista_padres[i]
            lista_hijos[i + 1] = lista_padres[i + 1]
            continue

        # Nombramos los padres para facilidad de uso
        padre_1 = lista_padres[i]
        padre_2 = lista_padres[i + 1]

        # Elegimos dos puntos de corte aleatorios
        punto_corte_1, punto_corte_2 = sorted(
            np.random.choice(tam_individuo, 2, replace=False)
        )

        # Inicializamos los hijos
        hijo_1 = np.full_like(padre_1, -1)
        hijo_2 = np.full_like(padre_2, -1)

        # El intervalo entre los puntos de corte es pasado directamente a los hijos
        hijo_2[punto_corte_1:punto_corte_2] = padre_2[punto_corte_1:punto_corte_2]
        hijo_1[punto_corte_1:punto_corte_2] = padre_1[punto_corte_1:punto_corte_2]

        # Obtenemos la lista de los números de cada padre a partir del punto final de corte ordenados por aparición
        lista_padre_1 = np.concatenate(
            [padre_1[punto_corte_2:], padre_1[:punto_corte_2]]
        )
        lista_padre_2 = np.concatenate(
            [padre_2[punto_corte_2:], padre_2[:punto_corte_2]]
        )

        # Eliminamos los números que ya están en el hijo contrario
        lista_padre_1 = lista_padre_1[~np.isin(lista_padre_1, hijo_2)]
        lista_padre_2 = lista_padre_2[~np.isin(lista_padre_2, hijo_1)]

        # Completamos los hijos con los números que faltan
        hijo_1[hijo_1 == -1] = lista_padre_2
        hijo_2[hijo_2 == -1] = lista_padre_1

        # Comprobamos que los hijos resultantes sean válidos y los añadimos a la lista de hijos
        # Si no son válidos, añadimos los padres a la lista de hijos
        if aptitud(hijo_1, matriz_adyacencia) == float("inf"):
            lista_hijos[i] = padre_1
        else:
            lista_hijos[i] = hijo_1

        if aptitud(hijo_2, matriz_adyacencia) == float("inf"):
            lista_hijos[i + 1] = padre_2
        else:
            lista_hijos[i + 1] = hijo_2

    return lista_hijos


def crossover_cycle_optimizado(
    lista_padres: np.ndarray,
    aptitud: Callable,
    matriz_adyacencia: np.ndarray,
    probabilidad: float,
) -> np.ndarray:
    """
    Realiza el cycle crossover según se explica en https://www.hindawi.com/journals/cin/2017/7430125/
    Resumidamente, se eligen padres de 2 en 2 y cada par produce 2 hijos.

    :param lista_padres: Array de NumPy de todos los padres a cruzar.
    :param aptitud: Función de aptitud.
    :param matriz_adyacencia: Array de NumPy que representa las distancias entre los nodos.
    :param probabilidad: Probabilidad de que se realice el crossover.
    :return: Array de NumPy que representa la lista de hijos.
    """
    num_padres, tam_individuo = lista_padres.shape

    # Inicializamos la matriz de hijos
    lista_hijos = np.zeros_like(lista_padres)

    # Iteramos sobre los padres de 2 en 2
    for i in range(0, num_padres, 2):
        if np.random.rand() > probabilidad:
            lista_hijos[i] = lista_padres[i]
            lista_hijos[i + 1] = lista_padres[i + 1]
            continue

        # Nombramos los padres para facilidad de uso
        padre_1 = lista_padres[i]
        padre_2 = lista_padres[i + 1]

        # Inicializamos los hijos
        hijo_1 = np.full_like(padre_1, -1)
        hijo_2 = np.full_like(padre_2, -1)

        # Cada hijo tendrá como base el padre con mismo número
        # Hacemos el primer ciclo de cada hijo
        j = 0
        while hijo_1[j] == -1:
            hijo_1[j] = padre_1[j]
            j = np.where(padre_2 == padre_1[j])[0][0]

        j = 0
        while hijo_2[j] == -1:
            hijo_2[j] = padre_2[j]
            j = np.where(padre_1 == padre_2[j])[0][0]

        # Completamos los hijos con los números que faltan. Serán del padre contrario
        hijo_1[hijo_1 == -1] = padre_2[hijo_1 == -1]
        hijo_2[hijo_2 == -1] = padre_1[hijo_2 == -1]

        # Comprobamos que los hijos resultantes sean válidos y los añadimos a la lista de hijos
        # Si no son válidos, añadimos los padres a la lista de hijos
        if aptitud(hijo_1, matriz_adyacencia) == float("inf"):
            lista_hijos[i] = padre_1
        else:
            lista_hijos[i] = hijo_1

        if aptitud(hijo_2, matriz_adyacencia) == float("inf"):
            lista_hijos[i + 1] = padre_2
        else:
            lista_hijos[i + 1] = hijo_2

    return lista_hijos


def elitismo_optimizado(
    poblacion: np.ndarray,
    num_elitismo: int,
    aptitud: Callable,
    matriz_adyacencia: np.ndarray,
) -> np.ndarray:
    """
    Selecciona los mejores individuos de la población utilizando NumPy para optimizar el proceso.

    :param poblacion: Array de NumPy que representa la población de individuos.
    :param num_elitismo: Número de individuos a seleccionar.
    :param aptitud: Función de aptitud.
    :param matriz_adyacencia: Array de NumPy que representa las distancias entre los nodos.
    :return: Array de NumPy que representa los individuos seleccionados.
    """
    # Obtenemos los índices de la población ordenada por aptitud
    indices_ordenados = np.argsort(
        [aptitud(individuo, matriz_adyacencia) for individuo in poblacion]
    )

    # Seleccionamos los num_elitismo primeros individuos
    elitismo_seleccionado = poblacion[indices_ordenados[:num_elitismo]]

    return elitismo_seleccionado


def seleccionar_torneo_optimizado(
    poblacion: np.ndarray,
    participantes: int,
    aptitud: Callable,
    matriz_adyacencia: np.ndarray,
    cantidad: int = None,
) -> np.ndarray:
    """
    Selecciona los mejores individuos de la población utilizando el método de torneo con NumPy para optimizar el proceso.

    :param poblacion: Array de NumPy que representa la población de individuos.
    :param participantes: Número de participantes en cada torneo.
    :param aptitud: Función de aptitud.
    :param matriz_adyacencia: Array de NumPy que representa las distancias entre los nodos.
    :param cantidad: Número de individuos a seleccionar.
    :return: Array de NumPy que representa los individuos seleccionados.
    """
    # Por defecto, seleccionamos la misma cantidad de individuos que hay en la población
    if cantidad is None:
        cantidad = len(poblacion)

    # Generamos índices aleatorios para los participantes en cada torneo
    indices_participantes = np.random.choice(
        len(poblacion), size=(cantidad, participantes), replace=True
    )

    # Evaluamos la aptitud de los participantes para cada torneo
    aptitudes_torneo = np.array(
        [
            [aptitud(poblacion[i], matriz_adyacencia) for i in torneo]
            for torneo in indices_participantes
        ]
    )

    # Encontramos el índice del mínimo en cada torneo
    indices_ganadores = np.argmin(aptitudes_torneo, axis=1)

    # Seleccionamos los individuos ganadores
    seleccionados = poblacion[
        indices_participantes[np.arange(cantidad), indices_ganadores]
    ]

    return seleccionados


def ejecutar_ejemplo_viajante(
    dibujar: bool = False, verbose: bool = True, parada_en_media=False
):
    # Ejecución de ejemplo

    municipios, distancias = leer_distancias_optimizada(
        "Viajante/Datos/matriz6.txt", "Viajante/Datos/pueblos6.txt"
    )
    
    if verbose:
        print("Municipios leídos.")

    poblacion = crear_poblacion_optimizada(
        len(municipios),
        NUM_INDIVIDUOS,
        aptitud_viajante_optimizada,
        distancias,
        False,
        verbose,
    )

    distancias_iteraciones = []
    distancias_medias = []

    for i in range(NUM_ITERACIONES):
        # Seleccionamos los individuos por torneo
        seleccionados = seleccionar_torneo_optimizado(
            poblacion, PARTICIPANTES_TORNEO, aptitud_viajante_optimizada, distancias
        )

        # Cruzamos los seleccionados
        hijos = crossover_partially_mapped_optimizado(
            seleccionados, aptitud_viajante_optimizada, distancias, PROB_CRUZAMIENTO
        )

        # Mutamos los hijos
        for hijo in hijos:
            if np.random.rand() < PROB_MUTACION:
                hijo = mutar_optimizada(hijo)

        # Elitismo
        poblacion = elitismo_optimizado(
            np.concatenate((poblacion, hijos)),
            len(municipios),
            aptitud_viajante_optimizada,
            distancias,
        )

        # Guardamos la distancia del mejor individuo
        distancias_iteraciones.append(
            aptitud_viajante_optimizada(poblacion[0], distancias, True)
        )

        # Guardamos la distancia media de la población
        distancias_medias.append(
            np.mean(
                [
                    aptitud_viajante_optimizada(individuo, distancias, True)
                    for individuo in poblacion
                ]
            )
        )

        if verbose:
            # Numero de iteración
            print("Iteración {i}".format(i=i))
            # Distancia del mejor individuo
            print(
                "Distancia mejor individuo: {dist}".format(
                    dist=distancias_iteraciones[-1]
                )
            )
            # Distancia media
            print("Distancia media: {dist}".format(dist=distancias_medias[-1]))

        if (
            parada_en_media
            and i > 10
            and distancias_medias[-1] == distancias_medias[-10]
        ):
            break

    # Ordenamos los individuos por aptitud
    poblacion = poblacion[
        np.argsort(
            [aptitud_viajante_optimizada(x, distancias, True) for x in poblacion]
        )
    ]
    mejor_distancia = aptitud_viajante_optimizada(poblacion[0], distancias, True)
    print("Mejor distancia: ", mejor_distancia)

    if dibujar:
        # Mostramos la evolución de la distancia con matplotlib
        plt.plot(distancias_iteraciones, label="Distancia mejor individuo")
        plt.plot(distancias_medias, label="Distancia media")
        # Añadir un recuadro con la menor distancia
        plt.text(0, 0, "Mejor distancia: {dist}".format(dist=mejor_distancia))
        plt.legend()
        plt.show()
        print(poblacion[0])
        print(distancias_iteraciones[-1])

    if verbose:
        # Imprimimos el recorrido con los nombres de los municipios
        for i in poblacion[0]:
            print(municipios[i])

    return distancias_iteraciones, distancias_medias


if __name__ == "__main__":
    mejores_aptitudes = []
    distancias_medias = []

    for i in range(1):
        aptitudes_mejor_individuo, distancias_media = ejecutar_ejemplo_viajante(
            dibujar=True, verbose=True, parada_en_media=True
        )
        mejores_aptitudes.append(aptitudes_mejor_individuo)
        distancias_medias.append(distancias_media)

    # Mostramos la evolución de la distancia del mejor individuo con matplotlib
    for i, aptitudes in enumerate(mejores_aptitudes):
        plt.plot(aptitudes, label="Mejor individuo {i}".format(i=i))
    plt.title("Evolución de la distancia del mejor individuo")
    plt.xlabel("Iteración")
    plt.ylabel("Distancia")
    plt.legend()
    plt.show()

    # Mostramos la evolución de la distancia media y del mejor individuo con matplotlib
    plt.clf()
    for i, (distancias_media, aptitudes_mejor_individuo) in enumerate(
        zip(distancias_medias, mejores_aptitudes)
    ):
        plt.plot(distancias_media, label="Media {i}".format(i=i))
        plt.plot(aptitudes_mejor_individuo, label="Mejor individuo {i}".format(i=i))
    plt.title("Evolución de la distancia media y del mejor individuo")
    plt.xlabel("Iteración")
    plt.ylabel("Distancia")
    plt.legend()
    plt.show()
