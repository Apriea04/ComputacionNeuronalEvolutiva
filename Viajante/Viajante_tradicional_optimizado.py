import random
from typing import Callable
import matplotlib.pyplot as plt
import numpy as np

# Todo este código está pensado desde el principio para minimizar la función de aptitud

# Código generado por IA


def leer_distancias_optimizada(path_distancias: str) -> np.ndarray:
    """
    Lee una matriz de distancias de un fichero de texto utilizando NumPy para optimizar el proceso.
    Si se proporciona un path_nombres, lee los nombres desde ese archivo; de lo contrario, asume que
    están en el mismo archivo que las distancias, en líneas precedidas por nombres entrecomillados.

    :param path_distancias: Ruta del fichero de distancias.
    :param path_nombres: Ruta al fichero que contiene los nombres (opcional).
    :return: Tuple que contiene una lista de nombres y un array de distancias de NumPy.
    """
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

    return distancias


def aptitud_viajante(individuo: np.ndarray, matriz_adyacencia: np.ndarray) -> float:
    """Devuelve la aptitud de un individuo. Se define como la suma de costes (distancias) de recorrer el camino que indica el individuo.
    Elementos a tener en cuenta para el cálculo de la aptitud:
    - El viajante tiene ubicación de salida fija, el final puede ser cualquier población.

    :param individuo: Array de enteros que representa el camino.
    :param matriz_adyacencia: Matriz de adyacencia que representa las distancias entre los nodos. Los valores de las coordenadas (i,i) corresponden al coste de estar en la población correspondiente.
    :return: Valor de aptitud del individuo.
    """
    # Sumamos el coste de para llegar a la población actual desde la anterior
    distancias = matriz_adyacencia[individuo[:-1], individuo[1:]]

    # Comprobamos si hay conexiones entre las poblaciones
    if np.any(distancias == 0):
        return float("inf")

    aptitud = np.sum(distancias)

    # SUMAMOS LA DISTANCIA DE VUELTA AL PUNTO DE PARTIDA
    aptitud += matriz_adyacencia[individuo[-1], individuo[0]]

    return aptitud


def crear_poblacion_optimizada(
    num_poblaciones: int,
    tam_poblacion: int,
    aptitud: Callable,
    matriz_adyacencia: np.ndarray,
    verbose: bool = False,
    unicos: bool = True,
) -> np.ndarray:
    """
    Crea una población de individuos utilizando NumPy para optimizar la generación.

    :param num_poblaciones: Número de poblaciones.
    :param tam_poblacion: Tamaño de la población.
    :param aptitud: Función de aptitud.
    :param matriz_adyacencia: Array de NumPy que representa las distancias entre los nodos.
    :param verbose: Si es True, imprime información del progreso.
    :return: Array de NumPy que representa individuos.
    """
    poblacion = np.zeros((tam_poblacion, num_poblaciones), dtype=int)

    for i in range(tam_poblacion):
        individuo = np.random.permutation(num_poblaciones)

        # Mientras el individuo no sea viable, generamos uno nuevo
        while aptitud(individuo, matriz_adyacencia) == float("inf"):
            individuo = np.random.shuffle(num_poblaciones)

        if unicos:
            while any(np.array_equal(individuo, p) for p in poblacion[:i]):
                individuo = np.random.shuffle(num_poblaciones)

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
    Muta un individuo utilizando NumPy para optimizar la mutación.
    Puede que el camino resultante no sea válido.

    :param individuo: Array de NumPy de enteros que representa el camino.
    :return: Array de NumPy de enteros que representa el camino mutado.
    """
    # Copiamos el individuo
    mutado = individuo.copy()

    # Elegimos dos posiciones aleatorias
    posiciones = np.random.choice(len(mutado), size=2, replace=False)

    # Intercambiamos los valores de las posiciones
    mutado[posiciones[0]], mutado[posiciones[1]] = (
        mutado[posiciones[1]],
        mutado[posiciones[0]],
    )

    return mutado


def crossover_partially_mapped_optimizado(
    lista_padres: np.ndarray,
    aptitud: Callable,
    matriz_adyacencia: np.ndarray,
    probabilidad: float,
) -> np.ndarray:
    """Realiza el crossover partially mapped según se explica en https://www.hindawi.com/journals/cin/2017/7430125/
    Resumidamente, se van eligiendo padres en orden y de 2 en 2.
    Cada 2 padres producen 2 hijos.
    :param lista_padres: Array de todos los padres a cruzar.
    :param aptitud: Función de aptitud.
    :param matriz_adyacencia: Matriz de adyacencia que representa las distancias entre los nodos.
    :param probabilidad: Probabilidad de que se realice el crossover.
    :return: Array de hijos.
    """
    num_padres = len(lista_padres)
    hijos = np.empty_like(lista_padres)

    for i in range(0, num_padres, 2):
        padre1, padre2 = lista_padres[i], lista_padres[(i + 1) % num_padres]

        if np.random.random() < probabilidad:
            punto1, punto2 = sorted(
                np.random.choice(range(len(padre1)), 2, replace=False)
            )
            hijo1, hijo2 = (
                padre1.copy(),
                padre2.copy(),
            )  # Copias de los padres para empezar

            # Mapeo para el hijo 1 y el hijo 2
            mapeo1, mapeo2 = {}, {}

            for j in range(punto1, punto2 + 1):
                gen1, gen2 = padre1[j], padre2[j]
                hijo1[j], hijo2[j] = gen2, gen1
                mapeo1[gen2], mapeo2[gen1] = gen1, gen2

            # Resolución de conflictos para ambos hijos
            for hijo, mapeo in zip((hijo1, hijo2), (mapeo1, mapeo2)):
                for k in range(len(hijo)):
                    if not (punto1 <= k <= punto2):
                        while hijo[k] in mapeo:
                            hijo[k] = mapeo[hijo[k]]

            hijos[i], hijos[i + 1] = hijo1, hijo2
        else:
            # Si no hay crossover, se agregan los padres originales
            hijos[i], hijos[i + 1] = padre1, padre2

    return hijos


def crossover_order_optimizado(
    lista_padres: np.ndarray,
    aptitud: Callable,
    matriz_adyacencia: np.ndarray,
    probabilidad: float,
) -> np.ndarray:
    """
    Realiza el order crossover optimizado utilizando NumPy para mejorar la eficiencia.

    :param lista_padres: Array de NumPy que contiene los padres.
    :param aptitud: Función de aptitud.
    :param matriz_adyacencia: Array de NumPy que representa las distancias entre los nodos.
    :param probabilidad: Probabilidad de aplicar el crossover.
    :return: Array de NumPy que contiene los hijos.
    """
    hijos = np.empty_like(lista_padres)

    num_padres = len(lista_padres)
    for i in range(0, num_padres, 2):
        padre1 = lista_padres[i]
        padre2 = lista_padres[
            (i + 1) % num_padres
        ]  # Asegura un crossover circular entre el último y el primer padre

        if random.random() < probabilidad:
            # Selecciona dos puntos de crossover aleatorios
            punto1, punto2 = sorted(random.sample(range(len(padre1)), 2))

            # Genera los hijos inicialmente vacíos
            hijo1, hijo2 = (
                np.full_like(padre1, fill_value=-1),
                np.full_like(padre2, fill_value=-1),
            )

            # Paso 1: Copia la sección del padre a los hijos
            hijo1[punto1 : punto2 + 1] = padre1[punto1 : punto2 + 1]
            hijo2[punto1 : punto2 + 1] = padre2[punto1 : punto2 + 1]

            # Paso 2: Completa los hijos con los genes del otro padre, manteniendo el orden y sin duplicar
            def completar_hijo(hijo, padre_donor):
                posiciones_vacias = np.where(hijo == -1)[0]
                posicion_actual = (punto2 + 1) % len(padre1)
                for gen in padre_donor:
                    if gen not in hijo:
                        hijo[posiciones_vacias[0]] = gen
                        posiciones_vacias = np.roll(posiciones_vacias, shift=-1)
                        posicion_actual = (posicion_actual + 1) % len(padre1)

            completar_hijo(
                hijo1, np.concatenate((padre2[punto2 + 1 :], padre2[: punto2 + 1]))
            )
            completar_hijo(
                hijo2, np.concatenate((padre1[punto2 + 1 :], padre1[: punto2 + 1]))
            )

            hijos[i] = hijo1
            hijos[i + 1] = hijo2
        else:
            # Si no hay crossover, solo copia los padres a la nueva generación
            hijos[i] = padre1
            hijos[i + 1] = padre2

    return hijos


def crossover_cycle_optimizado(
    lista_padres: np.ndarray,
    aptitud: Callable,
    matriz_adyacencia: np.ndarray,
    probabilidad: float,
) -> np.ndarray:
    """
    Realiza el order crossover optimizado utilizando NumPy para mejorar la eficiencia.

    :param lista_padres: Array de NumPy que contiene los padres.
    :param aptitud: Función de aptitud.
    :param matriz_adyacencia: Array de NumPy que representa las distancias entre los nodos.
    :param probabilidad: Probabilidad de aplicar el crossover.
    :return: Array de NumPy que contiene los hijos.
    """
    # Inicializamos el array de hijos
    lista_hijos = np.empty_like(lista_padres)

    # Iteramos sobre los padres de 2 en 2
    for i in range(0, len(lista_padres), 2):
        if random.random() > probabilidad:
            lista_hijos[i] = lista_padres[i]
            lista_hijos[i + 1] = lista_padres[i + 1]
            continue

        # Nombramos los padres para facilidad de uso
        padre_1 = lista_padres[i]
        padre_2 = lista_padres[i + 1]

        # Inicializamos los hijos
        hijo_1 = np.full_like(padre_1, fill_value=-1)
        hijo_2 = np.full_like(padre_2, fill_value=-1)

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
        hijo_1[hijo_1 == -1] = padre_2[~np.isin(padre_2, hijo_1)]
        hijo_2[hijo_2 == -1] = padre_1[~np.isin(padre_1, hijo_2)]

        # Comprobamos que los hijos resultantes sean válidos y los añadimos al array de hijos
        # Si no son válidos, añadimos los padres al array de hijos
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
    Selecciona los mejores individuos de la población utilizando NumPy para mejorar la eficiencia.

    :param poblacion: Array de NumPy que contiene los individuos.
    :param num_elitismo: Número de individuos a seleccionar.
    :param aptitud: Función de aptitud.
    :param matriz_adyacencia: Array de NumPy que representa las distancias entre los nodos.
    :return: Array de NumPy que contiene los individuos seleccionados.
    """
    # Ordenamos la población por aptitud utilizando np.argsort
    indices_ordenados = np.argsort(
        [aptitud(individuo, matriz_adyacencia) for individuo in poblacion]
    )

    # Devolvemos los num_elitismo primeros individuos utilizando la indexación avanzada de NumPy
    return poblacion[indices_ordenados[:num_elitismo]]


def seleccionar_torneo_optimizado(
    poblacion: np.ndarray,
    participantes: int,
    aptitud: Callable,
    matriz_adyacencia: np.ndarray,
    cantidad: int = None,
) -> np.ndarray:
    """
    Selecciona los mejores individuos de la población utilizando un enfoque de torneo y NumPy para mejorar la eficiencia.

    :param poblacion: Array de NumPy que contiene los individuos.
    :param participantes: Número de participantes en cada torneo.
    :param aptitud: Función de aptitud.
    :param matriz_adyacencia: Array de NumPy que representa las distancias entre los nodos.
    :param cantidad: Número de individuos a seleccionar.
    :return: Array de NumPy que contiene los individuos seleccionados.
    """
    # Por defecto, seleccionamos la misma cantidad de individuos que hay en la población
    if cantidad is None:
        cantidad = len(poblacion)

    # Inicializamos la lista de seleccionados
    seleccionados = np.empty((0, poblacion.shape[1]), dtype=poblacion.dtype)

    # Iteramos hasta que tengamos la cantidad de seleccionados que queremos
    while len(seleccionados) < cantidad:
        # Elegimos participantes al azar
        indices_participantes = np.random.choice(
            len(poblacion), size=participantes, replace=False
        )
        participantes_elegidos = poblacion[indices_participantes]

        # Calculamos la aptitud de cada participante
        aptitudes = np.array(
            [
                aptitud(individuo, matriz_adyacencia)
                for individuo in participantes_elegidos
            ]
        )

        # Elegimos el mejor de los participantes utilizando np.argmin
        indice_seleccionado = np.argmin(aptitudes)
        seleccionado = participantes_elegidos[indice_seleccionado]

        # Añadimos el seleccionado a la lista de seleccionados
        seleccionados = np.vstack((seleccionados, seleccionado))

    return seleccionados


def seleccionar_ruleta_pesos_optimizado(
    poblacion: np.ndarray,
    aptitud: Callable,
    matriz_adyacencia: np.ndarray,
    cantidad: int = None,
) -> np.ndarray:
    """
    Selecciona individuos de la población dándole más probabilidad a aquellos con menor aptitud.

    :param poblacion: Array de NumPy que contiene los individuos.
    :param aptitud: Función de aptitud.
    :param matriz_adyacencia: Array de NumPy que representa las distancias entre los nodos.
    :param cantidad: Número de individuos a seleccionar.
    :return: Array de NumPy que contiene los individuos seleccionados.
    """
    # Calculamos las aptitudes inversas
    aptitudes = np.array(
        [aptitud(individuo, matriz_adyacencia) for individuo in poblacion]
    )
    aptitudes_inversas = np.where(aptitudes > 0, 1 / aptitudes, float("inf"))

    # Calculamos el total de las aptitudes inversas para normalizar después
    suma_inversa = np.sum(aptitudes_inversas)

    # Calculamos las probabilidades acumuladas utilizando NumPy
    probabilidades_acumuladas = np.cumsum(aptitudes_inversas / suma_inversa)

    # Por defecto, seleccionamos la misma cantidad de individuos que hay en la población si no se especifica una cantidad
    if cantidad is None:
        cantidad = len(poblacion)

    # Generamos números aleatorios de manera vectorizada
    numeros_aleatorios = np.random.rand(cantidad)

    # Utilizamos la búsqueda binaria para encontrar los índices de los individuos seleccionados
    indices_seleccionados = np.searchsorted(
        probabilidades_acumuladas, numeros_aleatorios
    )

    # Seleccionamos los individuos a partir de los índices encontrados
    seleccionados = poblacion[indices_seleccionados]

    return seleccionados


def crossover_edge_recombination_optimizado(
    lista_padres: np.ndarray,
    aptitud: Callable,
    matriz_adyacencia: np.ndarray,
    probabilidad: float,
) -> np.ndarray:
    """
    Realiza el edge recombination crossover optimizado utilizando NumPy para mejorar la eficiencia.

    :param lista_padres: Array de NumPy que contiene los padres.
    :param aptitud: Función de aptitud.
    :param matriz_adyacencia: Array de NumPy que representa las distancias entre los nodos.
    :param probabilidad: Probabilidad de aplicar el crossover.
    :return: Array de NumPy que contiene los hijos.
    """
    # TODO: hacerlo más eficiente sin listas
    lista_hijos = []

    for i in range(0, len(lista_padres), 2):
        if np.random.rand() > probabilidad or i + 1 >= len(lista_padres):
            lista_hijos.append(lista_padres[i])
            if i + 1 < len(lista_padres):
                lista_hijos.append(lista_padres[i + 1])
            continue

        padre_1 = lista_padres[i]
        padre_2 = lista_padres[i + 1]

        # Creamos el diccionario de vecinos
        vecinos = {}
        for padre in [padre_1, padre_2]:
            for j in range(len(padre)):
                if padre[j] not in vecinos:
                    vecinos[padre[j]] = set()
                if j > 0:
                    vecinos[padre[j]].add(padre[j - 1])
                if j < len(padre) - 1:
                    vecinos[padre[j]].add(padre[j + 1])

        # Inicializamos el hijo con el primer valor de uno de los padres
        hijo = np.random.choice([padre_1[0], padre_2[0]], 1)

        # Y eliminamos ese valor como vecino de todos los nodos
        for vecino in list(vecinos.keys()):
            if hijo[0] in vecinos[vecino]:
                vecinos[vecino].remove(hijo[0])

        # Mientras no hayamos completado el hijo
        while len(hijo) < len(padre_1):
            actual = hijo[-1]

            if vecinos[actual]:
                # Seleccionamos el vecino con menos vecinos adicionales
                al_min = list(vecinos[actual])
                x = min(
                    al_min, key=lambda k: len(vecinos[k])
                )  # min de una lista con un solo elemento falla, ya que k no es clave del diccionario al haber sido borrado por ser vacío en la iteración anterior
            else:
                # Escogemos un elemento no incluido en el hijo al azar
                x = np.random.choice(
                    [elem for elem in vecinos.keys() if elem not in hijo]
                )
            hijo = np.append(hijo, x)

            # Eliminamos el elemento de las clases de vecinos
            del vecinos[actual]

            # Eliminamos el elemento actual de las listas de vecinos
            for vecino in list(vecinos.keys()):
                if x in vecinos[vecino]:
                    vecinos[vecino].remove(x)
                # Si la lista de vecinos está vacía, eliminamos el vecino del diccionario

        lista_hijos.append(hijo)

    return np.array(lista_hijos)


def crossover_pdf_optimizado(lista_padres: np.ndarray, aptitud: Callable, matriz_adyacencia: np.ndarray, probabilidad: float) -> np.ndarray:
    """Realiza el crossover según se explica en el PDF proporcionado por el profesor."""
    hijos = []
    num_padres = lista_padres.shape[0]
    for i in range(0, num_padres, 2):
        padre1 = lista_padres[i]
        padre2 = lista_padres[(i + 1) % num_padres]  # Asegura un crossover circular entre el último y el primer padre
        
        if random.random() < probabilidad:
            # Seleccionar dos puntos de cortes aleatorios
            punto1, punto2 = sorted(random.sample(range(len(padre1)), 2))
            
            seccion_padre1 = padre1[punto1:punto2]
            seccion_padre2 = padre2[punto1:punto2]
            
            resto_padre1 = np.setdiff1d(padre1, seccion_padre2)
            resto_padre2 = np.setdiff1d(padre2, seccion_padre1)
            
            # Inicializo a los hijos con -1
            hijo1 = np.full(padre1.shape, -1)
            hijo2 = np.full(padre2.shape, -1)
            
            hijo1[punto1:punto2] = seccion_padre2
            hijo2[punto1:punto2] = seccion_padre1
            
            resto1_idx = 0
            resto2_idx = 0
            for idx in range(len(padre1)):
                if not (punto1 <= idx < punto2):
                    hijo1[idx] = resto_padre1[resto1_idx]
                    hijo2[idx] = resto_padre2[resto2_idx]
                    resto1_idx += 1
                    resto2_idx += 1
                    
            hijos.append(hijo1)
            hijos.append(hijo2)
        else:
            # Si no hay crossover, solo copia los padres a la nueva generación
            hijos.append(padre1)
            hijos.append(padre2)
        
    return np.array(hijos)


def ejecutar_ejemplo_viajante_optimizado(
    dibujar: bool = False,
    verbose: bool = True,
    parada_en_media=False,
    parada_en_clones=False,
):
    if verbose:
        print("Municipios leídos.")
    poblacion = crear_poblacion_optimizada(
        len(MATRIZ[0]), NUM_INDIVIDUOS, aptitud_viajante, MATRIZ, verbose
    )

    if verbose:
        print("Población inicial:")
        for individuo in poblacion:
            print(individuo)

    distancias_iteraciones = []
    distancias_medias = []

    if verbose:
        print(
            "Cantidad de individuos distintos: ",
            len(set([aptitud_viajante(individuo, MATRIZ) for individuo in poblacion])),
        )

    for i in range(NUM_ITERACIONES):
        # Seleccionamos los individuos por torneo
        seleccionados = seleccionar_torneo_optimizado(
            poblacion,
            PARTICIPANTES_TORNEO,
            aptitud_viajante,
            MATRIZ,
            NUM_INDIVIDUOS * 2,
        )

        # Cruzamos los seleccionados
        hijos = crossover_pdf_optimizado(
            seleccionados, aptitud_viajante, MATRIZ, PROB_CRUZAMIENTO
        )

        # Mutamos los hijos
        for hijo in hijos:
            if np.random.rand() < PROB_MUTACION:
                hijo = mutar_optimizada(hijo)

        # Elitismo
        poblacion = elitismo_optimizado(
            np.concatenate((poblacion, hijos)), len(MATRIZ[0]), aptitud_viajante, MATRIZ
        )

        # Guardamos la distancia del mejor individuo
        distancias_iteraciones.append(aptitud_viajante(poblacion[0], MATRIZ))

        # Guardamos la distancia media de la población
        distancias_medias.append(
            np.mean([aptitud_viajante(individuo, MATRIZ) for individuo in poblacion])
        )

        if verbose:
            print("Iteración {i}".format(i=i))
            print(
                "Distancia mejor individuo: {dist}".format(
                    dist=distancias_iteraciones[-1]
                )
            )
            print("Distancia media: {dist}".format(dist=distancias_medias[-1]))
            print(
                "Cantidad de individuos distintos: ",
                len(
                    set(
                        [aptitud_viajante(individuo, MATRIZ) for individuo in poblacion]
                    )
                ),
            )

        if (
            parada_en_media
            and i > 10
            and distancias_medias[-1] == distancias_medias[-10]
        ):
            break

        if parada_en_clones:
            if (
                len(
                    set(
                        [aptitud_viajante(individuo, MATRIZ) for individuo in poblacion]
                    )
                )
                == 1
            ):
                break

    # Ordena la población según la aptitud
    indices_ordenados = np.argsort([aptitud_viajante(sol, MATRIZ) for sol in poblacion])
    # Reordena la población
    poblacion = poblacion[indices_ordenados]

    mejor_distancia = aptitud_viajante(poblacion[0], MATRIZ)

    if verbose:
        print("Mejor distancia: ", mejor_distancia)
        print("Mejor individuo: ", poblacion[0])

    if dibujar:
        plt.plot(distancias_iteraciones, label="Distancia mejor individuo")
        plt.plot(distancias_medias, label="Distancia media")
        plt.text(0, 0, "Mejor distancia: {dist}".format(dist=mejor_distancia))
        plt.legend()
        plt.show()
        print(poblacion[0])
        print(distancias_iteraciones[-1])

    return distancias_iteraciones, distancias_medias, poblacion[0]


# ----------------------------------------------------------------------
# Parámetros
NUM_ITERACIONES = 10000
PROB_MUTACION = 0.1
PROB_CRUZAMIENTO = 0.8
PARTICIPANTES_TORNEO = 2
NUM_INDIVIDUOS = 20
RUTA_MATRIZ = "Viajante/Datos/50_distancias.txt"
MATRIZ = leer_distancias_optimizada(RUTA_MATRIZ)
# ----------------------------------------------------------------------

if __name__ == "__main__":
    mejores_aptitudes = []
    distancias_medias = []
    mejores_individuos = []

    for i in range(10):
        apt, med, ind = ejecutar_ejemplo_viajante_optimizado(False, True, True, True)
        mejores_aptitudes.append(apt)
        distancias_medias.append(med)
        mejores_individuos.append(ind)

    # Mostramos el mejor individuo del total
    print(
        "Mejor individuo de todos: ",
        min(mejores_individuos, key=lambda x: aptitud_viajante(x, MATRIZ)),
    )
    print(
        "Mejor distancia de todos: ",
        aptitud_viajante(
            min(mejores_individuos, key=lambda x: aptitud_viajante(x, MATRIZ)), MATRIZ
        ),
    )

    if True:
        # Mostramos la evolución de la distancia con matplotlib
        for i in range(len(mejores_aptitudes)):
            plt.plot(mejores_aptitudes[i], label="Mejor individuo {i}".format(i=i))
        plt.legend()
        plt.show()

        plt.clf()
        # Dibujamos un grafico con la mejor distancia y la media de cada ejecución
        for i in range(len(distancias_medias)):
            plt.plot(mejores_aptitudes[i], label="Mejor individuo {i}".format(i=i))
        plt.legend()
        plt.show()
