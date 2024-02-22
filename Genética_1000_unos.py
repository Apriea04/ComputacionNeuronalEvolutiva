import random
import numpy as np


def mutar(
    poblacion: np.ndarray,
    probabilidad: float,
    max_genes: int = 1,
    verbose: bool = False,
) -> None:
    """Puede generar mutaciones en individuos de la población.
    :param poblacion: matriz de individuos
    :param probabilidad: probabilidad de mutación
    :param max_genes: número máximo de genes a mutar
    :param verbose: si se desea imprimir información sobre la mutación
    :return: None"""

    # Recorremos la población
    for cromosoma in poblacion:
        # Si sufre una mutación
        if random.uniform(0, 1) <= probabilidad:
            for _ in range(random.randint(1, max_genes)):
                # Elegimos el índice del gen a mutar
                indice_gen = random.randint(0, len(cromosoma) - 1)

                # Y lo mutamos
                if cromosoma[indice_gen] == 0:
                    cromosoma[indice_gen] = 1
                else:
                    cromosoma[indice_gen] = 0
            if verbose:
                print("Mutado:", cromosoma)


def aptitud(individuo: np.ndarray) -> float:
    """Calcula la aptitud de un individuo. En nuestro caso será la suma de sus genes.
    :param individuo: matriz de genes
    :return: aptitud del individuo"""
    individuo = np.array(individuo)
    return np.sum(individuo)


def crear_poblacion(n_individuos: int, n_genes: int) -> np.ndarray:
    """Crea una población de individuos con n_genes.
    :param n_individuos: número de individuos
    :param n_genes: número de genes
    :return: matriz de individuos"""
    poblacion = []
    for _ in range(n_individuos):
        cromosoma = []
        for _ in range(n_genes):
            cromosoma.append(random.randint(0, 1))
        poblacion.append(cromosoma)
        print(cromosoma)
    return np.array(poblacion)


# TODO: Hecho con Copilot para poder probar el código
def seleccionar_padres(poblacion: np.ndarray, n_padres: int) -> np.ndarray:
    """Selecciona n_padres de la población, siendo más probable que los individuos con
    mayor aptitud sean seleccionados.
    :param poblacion: matriz de individuos
    :param n_padres: número de padres a seleccionar
    :return: matriz de padres seleccionados"""
    poblacion = np.array(poblacion)
    padres = []
    for _ in range(n_padres):
        # Seleccionamos tres individuos al azar
        candidatos = random.sample(list(poblacion), 8)
        # Y nos quedamos con el que tenga mayor aptitud
        padres.append(max(candidatos, key=aptitud))
    return np.array(padres)


def seleccionar_ruleta(poblacion: np.ndarray, n: int) -> np.ndarray:
    """Selecciona n individuos de la población usando el método de la ruleta.
    :param poblacion: matriz de individuos
    :param n: número de individuos a seleccionar
    :return: matriz de individuos seleccionados"""

    aptitudes = [aptitud(individuo) for individuo in poblacion]
    poblacion_seleccionada = []
    aptitudes_acumuladas = [sum(aptitudes[: i + 1]) for i in range(len(aptitudes))]
    aptitud_total = aptitudes_acumuladas[-1]

    # Para cada progenitor a seleccionar
    for _ in range(n):
        valor = random.randint(0, aptitud_total)

        seleccionado = 0

        while not aptitudes_acumuladas[seleccionado] >= valor:
            seleccionado += 1
        poblacion_seleccionada.append(poblacion[seleccionado])
    return np.array(poblacion_seleccionada)


def crossover(
    progenitores: np.ndarray,
    tipo: int = 1,
    num_progenitores_involucrados: int = 2,
    verbose: bool = False,
) -> np.ndarray:
    """Produce un nuevo individuo a partir de algún tipo de crossover, involucrando a
    num_progenitores_involucrados de entre *progenitores.
    :param poblacion: matriz de individuos
    :param progenitores: matriz de progenitores
    :param tipo: tipo de crossover a realizar
    :param num_progenitores_involucrados: número de progenitores a involucrar
    :param verbose: si se desea imprimir información sobre el crossover
    :return: matriz de individuos con el nuevo individuo agregado"""

    match tipo:
        # 1: Corte sencillo
        case 1:
            indices_corte = [
                random.randint(0, len(progenitores[0]) - 1)
                for _ in range(num_progenitores_involucrados - 1)
            ]
            if verbose:
                print("Índices de corte:", indices_corte)
            # Obtengo solo a los progenitores involucrados entre los disponibles
            progenitores_involucrados = random.sample(
                list(progenitores), num_progenitores_involucrados
            )
            if verbose:
                print("Progenitores involucrados:", progenitores_involucrados)
            # Declaro al sucesor
            sucesor = []
            progenitor_actual = 0
            for i in range(len(progenitores[0])):
                sucesor.append(progenitores_involucrados[progenitor_actual][i])
                if i in indices_corte:
                    progenitor_actual += 1
            if verbose:
                print("Sucesor:", sucesor)

            return np.array(sucesor)


def run():
    # Creo una población de 12 individuos
    poblacion = crear_poblacion(12, 100)

    # Guardo la media inicial en medias
    sumas = np.sum(poblacion, 1)
    print(sumas)
    medias = [np.mean(sumas)]
    print(medias)
    mejores = np.max(sumas)
    print(mejores)

    for _ in range(150):
        nueva_generacion = []
        for _ in range(12):
            crossover(nueva_generacion, 1, 2, False, seleccionar_padres(poblacion, 2))
        mutar(nueva_generacion, 0.01)
        poblacion = nueva_generacion
        sumas = np.sum(poblacion, 1)
        medias.append(np.mean(sumas))
        mejores = np.max(sumas)
