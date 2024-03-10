import random
from typing import Callable, List
import matplotlib.pyplot as plt

# Todo este código está pensado desde el principio para minimizar la función de aptitud


def leer_distancias(path_distancias: str, path_nombres: str = None) -> tuple:
    """Lee una matriz distancias de un fichero de texto.
    :param path_distancias: Ruta del fichero. Puede contener una matriz separando los valores por espacios, o los nombres de los municipios entrecomillados precediendo cada uno a la fila de distancias correspondiente.
    :param path_nombres: Ruta al fichero que contiene los nombres.
    :return: lista de nombres, matriz de distancias.
    """
    # Inicializamos una lista para los nombres de los municipios y una lista de listas para la matriz de números
    nombres_municipios = []
    matriz_numeros = []

    if path_nombres is None:
        # Abrimos el archivo para lectura
        with open(path_distancias, "r", encoding="utf-8") as archivo:
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
                matriz_numeros.append([float(numero) for numero in numeros])

    else:
        # Leemos el fichero de nombres
        with open(path_nombres, "r", encoding="utf-8") as archivo:
            nombres_municipios = [linea.strip() for linea in archivo]

        # Leemos el fichero de distancias
        with open(path_distancias, "r", encoding="utf-8") as archivo:
            for linea in archivo:
                matriz_numeros.append(
                    [float(numero) for numero in linea.strip().split()]
                )
    # Devolvemos los nombres y la matriz
    return nombres_municipios, matriz_numeros


def aptitud_viajante(individuo: list, matriz_adyacencia: list) -> float:
    """Devuelve la aptitud de un individuo. Se define como la suma de costes (distancias) de recorrer el camino que indica el individuo.

    :param individuo: Lista de enteros que representa el camino.
    :param matriz_adyacencia: Matriz de adyacencia que representa las distancias entre los nodos. Los valores de las coordenadas (i,i) corresponden al coste de estar en la población correspondiente.
    :return: Valor de aptitud del individuo.
    """
    aptitud = 0

    for i in range(1, len(individuo)):
        # Sumamos el coste de para llegar a la población actual desde la anterior
        distancia = matriz_adyacencia[individuo[i - 1]][individuo[i]]

        # Si la distancia es 0, es que no hay conexión entre las poblaciones. Luego el individuo no es válido
        if distancia == 0:
            return float("inf")

        aptitud += distancia
        
    # TODO: Decidir si dejar o no esto
    # SUMAMOS LA DISTANCIA DE VUELTA AL PUNTO DE PARTIDA
    aptitud += matriz_adyacencia[individuo[-1]][individuo[0]]

    return aptitud


def crear_poblacion(
    num_poblaciones: int,
    tam_poblacion: int,
    aptitud: Callable,
    matriz_adyacencia: list,
    verbose: bool = False,
) -> list:
    """Crea una población de individuos.
    :param num_poblaciones: Número de poblaciones.
    :param tam_poblacion: Tamaño de la población.
    :param aptitud: Función de aptitud.
    :param matriz_adyacencia: Matriz de adyacencia que representa las distancias entre los nodos.
    :param verbose: Si es True, imprime información del progreso.
    :return: Lista de listas que representan individuos.
    """
    # Creamos una lista de listas con el tamaño de la población
    poblacion = []
    while len(poblacion) < tam_poblacion:
        # Creamos un individuo aleatorio
        individuo = list(range(0, num_poblaciones))
        random.shuffle(individuo)

        # Solo añadimos el individuo si propone un recorrido viable
        while aptitud(individuo, matriz_adyacencia) == float("inf"):
            random.shuffle(individuo)
        poblacion.append(individuo)

        if verbose:
            print(
                "Creados {n} de {t} individuos".format(
                    n=len(poblacion), t=tam_poblacion
                ),
                end="\r",
            )
    if verbose:
        print()
    return poblacion


def mutar(individuo: list) -> list:
    """Muta un individuo. Puede que el camino resultante no sea válido.
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


def crossover_partially_mapped(
    lista_padres: list, aptitud: Callable, matriz_adyacencia: list, probabilidad: float
) -> list:
    """Realiza el crossover partially mapped según se explica en https://www.hindawi.com/journals/cin/2017/7430125/
    Resumidamente, se van eligiendo padres en orden y de 2 en 2.
    Cada 2 padres producen 2 hijos.
    :param lista_padres: Lista de todos los padres a cruzar.
    :param aptitud: Función de aptitud.
    :param matriz_adyacencia: Matriz de adyacencia que representa las distancias entre los nodos.
    :param probabilidad: Probabilidad de que se realice el crossover.
    :return: Lista de hijos.
    """
    num_padres = len(lista_padres)
    hijos = []

    for i in range(0, num_padres, 2):
        padre1, padre2 = lista_padres[i], lista_padres[(i + 1) % num_padres]

        if random.random() < probabilidad:
            punto1, punto2 = sorted(random.sample(range(len(padre1)), 2))
            hijo1, hijo2 = padre1[:], padre2[:]  # Copias de los padres para empezar

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

            hijos += [hijo1, hijo2]
        else:
            # Si no hay crossover, se agregan los padres originales
            hijos += [padre1, padre2]

    return hijos


def crossover_order(lista_padres: list, aptitud: Callable, matriz_adyacencia: list, probabilidad: float) -> list:
    """Realiza el order crossover."""
    hijos = []

    num_padres = len(lista_padres)
    for i in range(0, num_padres, 2):
        padre1 = lista_padres[i]
        padre2 = lista_padres[(i + 1) % num_padres]  # Asegura un crossover circular entre el último y el primer padre
        
        if random.random() < probabilidad:
            # Selecciona dos puntos de crossover aleatorios
            punto1, punto2 = sorted(random.sample(range(len(padre1)), 2))
            
            # Genera los hijos inicialmente vacíos
            hijo1, hijo2 = [None]*len(padre1), [None]*len(padre2)
            
            # Paso 1: Copia la sección del padre a los hijos
            hijo1[punto1:punto2+1] = padre1[punto1:punto2+1]
            hijo2[punto1:punto2+1] = padre2[punto1:punto2+1]
            
            # Paso 2: Completa los hijos con los genes del otro padre, manteniendo el orden y sin duplicar
            def completar_hijo(hijo, padre_donor):
                posicion_actual = (punto2 + 1) % len(padre1)
                for gen in padre_donor:
                    if gen not in hijo:
                        hijo[posicion_actual] = gen
                        posicion_actual = (posicion_actual + 1) % len(padre1)
            
            completar_hijo(hijo1, padre2[punto2+1:] + padre2[:punto2+1])
            completar_hijo(hijo2, padre1[punto2+1:] + padre1[:punto2+1])
            
            hijos.append(hijo1)
            hijos.append(hijo2)
        else:
            # Si no hay crossover, solo copia los padres a la nueva generación
            hijos.append(padre1[:])
            hijos.append(padre2[:])
    
    return hijos


def crossover_cycle(
    lista_padres: list, aptitud: Callable, matriz_adyacencia: list, probabilidad: float
) -> list:
    """Realiza el order crossover según se explica en https://www.hindawi.com/journals/cin/2017/7430125/
    Resumidamente, se van eligiendo padres en orden y de 2 en 2.
    Cada 2 padres producen 2 hijos.
    :param lista_padres: Lista de todos los padres a cruzar.
    :param aptitud: Función de aptitud.
    :param matriz_adyacencia: Matriz de adyacencia que representa las distancias entre los nodos.
    :param probabilidad: Probabilidad de que se realice el crossover.
    :return: Lista de hijos.
    """
    # Incializamos la lista de hijos
    lista_hijos = []

    # Iteramos sobre los padres de 2 en 2
    for i in range(0, len(lista_padres), 2):
        if random.random() > probabilidad:
            lista_hijos.append(lista_padres[i])
            lista_hijos.append(lista_padres[i + 1])
            continue

        # Nombramos los padres para facilidad de uso
        padre_1 = lista_padres[i]
        padre_2 = lista_padres[i + 1]

        # Inicializamos los hijos
        hijo_1 = [-1 for _ in range(len(padre_1))]
        hijo_2 = [-1 for _ in range(len(padre_1))]

        # Cada hijo tendrá como base el padre con mismo numero
        # Hacemos el primer ciclo de cada hijo
        j = 0
        while hijo_1[j] == -1:
            hijo_1[j] = padre_1[j]
            j = padre_2.index(padre_1[j])

        j = 0
        while hijo_2[j] == -1:
            hijo_2[j] = padre_2[j]
            j = padre_1.index(padre_2[j])

        # Completamos los hijos con los números que faltan. Serán del padre contrario
        for j in range(len(hijo_1)):
            if hijo_1[j] == -1:
                hijo_1[j] = padre_2[j]
            if hijo_2[j] == -1:
                hijo_2[j] = padre_1[j]

        # Comprobamos que los hijos resultantes sean válidos y los añadimos a la lista de hijos
        # Si no son válidos, añadimos los padres a la lista de hijos
        if aptitud(hijo_1, matriz_adyacencia) == float("inf"):
            lista_hijos.append(padre_1)
        else:
            lista_hijos.append(hijo_1)

        if aptitud(hijo_2, matriz_adyacencia) == float("inf"):
            lista_hijos.append(padre_2)
        else:
            lista_hijos.append(hijo_2)

    return lista_hijos


def elitismo(
    poblacion: list, num_elitismo: int, aptitud: Callable, matriz_adyacencia: list
) -> list:
    """Selecciona los mejores individuos de la población.
    :param poblacion: Lista de individuos.
    :param num_elitismo: Número de individuos a seleccionar.
    :param aptitud: Función de aptitud.
    :param matriz_adyacencia: Matriz de adyacencia que representa las distancias entre los nodos.
    :return: Lista de individuos seleccionados.
    """
    # Ordenamos la población por aptitud
    poblacion.sort(key=lambda x: aptitud(x, matriz_adyacencia))
    # Devolvemos los num_elitismo primeros individuos
    return poblacion[:num_elitismo]


def seleccionar_torneo(
    poblacion: list,
    participantes: int,
    aptitud: Callable,
    matriz_adyacencia: list,
    cantidad: int = None,
) -> list:
    """Selecciona los mejores individuos de la población.
    :param poblacion: Lista de individuos.
    :param participantes: Número de participantes en cada torneo.
    :param aptitud: Función de aptitud.
    :param matriz_adyacencia: Matriz de adyacencia que representa las distancias entre los nodos.
    :param cantidad: Número de individuos a seleccionar.
    :return: Lista de individuos seleccionados.
    """
    # Por defecto seleccionamos la misma cantidad de individuos que hay en la población
    if cantidad is None:
        cantidad = len(poblacion)
    # Inicializamos la lista de seleccionados
    seleccionados = []
    # Iteramos hasta que tengamos la cantidad de seleccionados que queremos
    while len(seleccionados) < cantidad:
        # Elegimos participantes al azar
        participantes_elegidos = random.sample(poblacion, participantes)
        # Elegimos el mejor de los participantes
        seleccionado = min(
            participantes_elegidos, key=lambda x: aptitud(x, matriz_adyacencia)
        )
        # Añadimos el seleccionado a la lista de seleccionados
        seleccionados.append(seleccionado)
    return seleccionados


def seleccionar_ruleta_pesos(
    poblacion: list,
    aptitud: Callable,
    matriz_adyacencia: list,
    cantidad: int = None,
) -> list:
    """Selecciona individuos de la población dándole más probabilidad a aquellos con menor aptitud.

    Parameters
    ----------
    poblacion: Lista de individuos.
    aptitud: Función de aptitud.
    matriz_adyacencia: Matriz de adyacencia que representa las distancias entre los nodos.
    cantidad: Número de individuos a seleccionar.

    Returns
    -------
    Lista de individuos seleccionados.
    """

    # Creamos la lista donde se almacenarán los inversos de los valores de aptitud
    aptitudes_inversas = []
    for individuo in poblacion:
        # Calculamos el inverso de la aptitud para que los de menor aptitud tengan mayor peso
        valor_aptitud = aptitud(individuo, matriz_adyacencia)
        aptitudes_inversas.append(
            1 / valor_aptitud if valor_aptitud > 0 else float("inf")
        )

    # Calculamos el total de las aptitudes inversas para normalizar después
    suma_inversa = sum(aptitudes_inversas)

    # Creamos la lista de probabilidades acumuladas a partir de las aptitudes inversas
    probabilidades_acumuladas = []
    suma_acumulada = 0
    for aptitud_inversa in aptitudes_inversas:
        probabilidad = aptitud_inversa / suma_inversa
        suma_acumulada += probabilidad
        probabilidades_acumuladas.append(suma_acumulada)

    # Por defecto seleccionamos la misma cantidad de individuos que hay en la población si no se especifica una cantidad
    if cantidad is None:
        cantidad = len(poblacion)

    # Inicializamos la lista de seleccionados
    seleccionados = []

    # Iteramos hasta que tengamos la cantidad de seleccionados que queremos
    while len(seleccionados) < cantidad:
        num_aleatorio = random.random()
        # Elegimos el primer individuo cuya probabilidad acumulada sea mayor que el número aleatorio
        for i, prob_acumulada in enumerate(probabilidades_acumuladas):
            if prob_acumulada > num_aleatorio:
                seleccionados.append(poblacion[i])
                break

    return seleccionados


def crossover_edge_recombination(
    lista_padres: list, aptitud: Callable, matriz_adyacencia: list, probabilidad: float
) -> list:
    """Realiza el edge recombination crossover según se explica en la página de Wikipedia y Rubicite mencionadas."""

    lista_hijos = []

    for i in range(0, len(lista_padres), 2):
        if random.random() > probabilidad or i + 1 >= len(lista_padres):
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
        hijo = random.sample([padre_1[0], padre_2[0]], 1)

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
                )  # min de una lista con un solo elemento falla, ya que k no es clave del diccionario al habe sido borrado por ser vacío en la iteración anterior
            else:
                # Escogemos un elemento no incluido en el hijo al azar
                x = random.choice([elem for elem in vecinos.keys() if elem not in hijo])
            hijo.append(x)

            # Eliminamos el elemento de las clases de vecinos
            del vecinos[actual]

            # Eliminamos el elemento actual de las listas de vecinos
            for vecino in list(vecinos.keys()):
                if x in vecinos[vecino]:
                    vecinos[vecino].remove(x)
                # Si la lista de vecinos está vacía, eliminamos el vecino del diccionario

        lista_hijos.append(hijo)

    return lista_hijos


def ejecutar_ejemplo_viajante(
    dibujar: bool = False, verbose: bool = True, parada_en_media=False
):
    # Ejecución de ejemplo

    if verbose:
        print("Municipios leídos.")
    poblacion = crear_poblacion(
        len(PUEBLOS), NUM_INDIVIDUOS, aptitud_viajante, MATRIZ, verbose
    )

    if verbose:
        print("Población inicial:")
        for individuo in poblacion:
            print(individuo)

    distancias_iteraciones = []
    distancias_medias = []

    for i in range(NUM_ITERACIONES):
        # Seleccionamos los individuos por torneo
        seleccionados = seleccionar_torneo(
            poblacion,
            PARTICIPANTES_TORNEO,
            aptitud_viajante,
            MATRIZ,
            NUM_INDIVIDUOS * 2,
        )

        # Cruzamos los seleccionados
        hijos = crossover_partially_mapped(
            seleccionados, aptitud_viajante, MATRIZ, PROB_CRUZAMIENTO
        )

        # Mutamos los hijos
        for hijo in hijos:
            if random.random() < PROB_MUTACION:
                hijo = mutar(hijo)

        # Elitismo
        poblacion = elitismo(
           poblacion + hijos, len(PUEBLOS), aptitud_viajante, MATRIZ
        )

        poblacion = hijos

        # Guardamos la distancia del mejor individuo
        distancias_iteraciones.append(aptitud_viajante(poblacion[0], MATRIZ))

        # Guardamos la distancia media de la población
        distancias_medias.append(
            sum(aptitud_viajante(individuo, MATRIZ) for individuo in poblacion)
            / len(poblacion)
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
    poblacion.sort(key=lambda x: aptitud_viajante(x, MATRIZ))
    mejor_distancia = aptitud_viajante(poblacion[0], MATRIZ)
    if verbose:
        print("Mejor distancia: ", mejor_distancia)
        print("Mejor individuo: ", poblacion[0])

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

    return distancias_iteraciones, distancias_medias, poblacion[0]

# ----------------------------------------------------------------------
# Parámetros
NUM_ITERACIONES = 10000
PROB_MUTACION = 0.1
PROB_CRUZAMIENTO = 0.8
PARTICIPANTES_TORNEO = 2
NUM_INDIVIDUOS = 100
RUTA_MATRIZ = "Viajante/Datos/matriz10.data"
RUTA_PUEBLOS = "Viajante/Datos/pueblos10.txt"
PUEBLOS, MATRIZ = leer_distancias(RUTA_MATRIZ, RUTA_PUEBLOS)
# ----------------------------------------------------------------------

if  __name__ == "__main__":
    mejores_aptitudes = []
    distancias_medias = []
    mejores_individuos = []
    
    for i in range(1):
        apt, med, ind = ejecutar_ejemplo_viajante(True, True, True)
        mejores_aptitudes.append(apt)
        distancias_medias.append(med)
        mejores_individuos.append(ind)
        
    # Mostramos el mejor individuo del total
    print("Mejor individuo de todos: ", min(mejores_individuos, key=lambda x: aptitud_viajante(x, MATRIZ)))
    print("Mejor distancia de todos: ", aptitud_viajante(min(mejores_individuos, key=lambda x: aptitud_viajante(x, MATRIZ)), MATRIZ))

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
