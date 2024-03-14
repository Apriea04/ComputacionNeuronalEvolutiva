import threading
from Viajante_tradicional_optimizado import (
    leer_distancias_optimizada,
    crear_poblacion_optimizada,
    aptitud_viajante,
    seleccionar_torneo_optimizado,
    crossover_partially_mapped_optimizado,
    crossover_edge_recombination_optimizado,
    crossover_cycle_optimizado,
    crossover_order_optimizado,
    crossover_pdf_optimizado,
    mutar_optimizada,
    elitismo_optimizado,
    mutar_mejorada_optimizada,
    elitismo_n_padres_optimizado,
    mutar_desordenado_optimizada,
)
from Datos.datos import leer_coordenadas
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing


def dibujar_individuo(
    individuo: np.ndarray,
    coordenadas: np.ndarray,
    graph_anterior=None,
    sleep_time=0.001,
):
    """Dibuja la ruta del mejor individuo y borra la ruta anterior."""
    ruta_x = [coordenadas[i][0] for i in individuo]
    ruta_y = [coordenadas[i][1] for i in individuo]
    graph = plt.plot(
        ruta_x + [ruta_x[0]],
        ruta_y + [ruta_y[0]],
        marker="o",
        linestyle="-",
        color="blue",
    )[0]

    # Borra la ruta anterior si se proporciona
    if graph_anterior is not None:
        graph_anterior.remove()  # Borra el grafo anterior

    plt.pause(sleep_time)  # Pausa para que se vea el cambio

    return graph


def ejecutar_ejemplo_viajante_optimizado(
    dibujar_evolucion: bool = False,
    verbose: bool = True,
    parada_en_media=True,
    elitismo: bool = True,
    parada_en_clones=False,
    plot_resultados_parciales: bool = True,
    cambio_de_mutacion: bool = False,
):
    # ----------------------------------------------------------------------
    # Parámetros
    NUM_ITERACIONES = 500
    MAX_MEDIAS_IGUALES = 10
    PROB_MUTACION = 0.1 #Visto
    PROB_CRUZAMIENTO = 0.4
    PARTICIPANTES_TORNEO = 2
    NUM_INDIVIDUOS = 100
    GENES_MUTAR = 2
    RUTA_COORDENADAS = "Viajante/Datos/50_coordenadas.txt"
    COORDENADAS, MATRIZ = leer_coordenadas(RUTA_COORDENADAS)
    # ----------------------------------------------------------------------

    if verbose:
        print("Municipios leídos.")
    poblacion = crear_poblacion_optimizada(
        len(MATRIZ[0]), NUM_INDIVIDUOS, aptitud_viajante, MATRIZ, verbose, True
    )

    if verbose:
        print("Población inicial:")
        for individuo in poblacion:
            print(individuo)

    if plot_resultados_parciales:
        plt.ion()  # turn on interactive mode
        plt.grid(color="salmon")
        plt.axvline(0, color="salmon")
        plt.axhline(0, color="salmon")
        plt.title("Mejor Individuo")
        plt.scatter(*zip(*COORDENADAS))

        grafico_nuevo = plt.plot(poblacion[0])[0]

    distancias_iteraciones = []
    distancias_medias = []

    for i in range(NUM_ITERACIONES):
        # Seleccionamos los individuos por torneo
        seleccionados = seleccionar_torneo_optimizado(
            poblacion,
            PARTICIPANTES_TORNEO,
            aptitud_viajante,
            MATRIZ,
            NUM_INDIVIDUOS,  # TODO: esto depende del crossover
        )

        # Cruzamos los seleccionados
        hijos = crossover_order_optimizado(
            seleccionados, aptitud_viajante, MATRIZ, PROB_CRUZAMIENTO
        )

        # Mutamos los hijos
        for hijo in hijos:
            if np.random.rand() < PROB_MUTACION:
                hijo = mutar_desordenado_optimizada(hijo)

        # Elitismo
        if elitismo:
            poblacion = elitismo_n_padres_optimizado(
                1,
                poblacion,
                hijos,
                NUM_INDIVIDUOS,
                aptitud_viajante,
                MATRIZ,
            )
            if False and elitismo:
                poblacion = elitismo_optimizado(
                    np.concatenate((poblacion, hijos)),
                    NUM_INDIVIDUOS,
                    aptitud_viajante,
                    MATRIZ,
                )

        else:
            poblacion = hijos

        if plot_resultados_parciales:
            grafico_nuevo = dibujar_individuo(
                poblacion[0], COORDENADAS, grafico_nuevo, 0.1
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

        if parada_en_media and i > MAX_MEDIAS_IGUALES:
            valor_media = distancias_medias[-1]
            salir = True
            for k in range(1, MAX_MEDIAS_IGUALES):
                if distancias_medias[-k] != valor_media:
                    salir = False
                    break
            if salir:
                break

        if (
            cambio_de_mutacion
            and i > MAX_MEDIAS_IGUALES // 2
            and distancias_medias[-1] == distancias_medias[-MAX_MEDIAS_IGUALES // 2]
        ):
            valor_media = distancias_medias[-1]
            cambiar_mutacion = True
            for k in range(1, MAX_MEDIAS_IGUALES // 2):
                if distancias_medias[-k] != valor_media:
                    cambiar_mutacion = False
                    break
            if cambiar_mutacion:
                print(
                    "#################################  Cambio de mutación  #################################"
                )
                PROB_MUTACION = 0.5
                GENES_MUTAR = 2

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

    if plot_resultados_parciales:
        plt.ioff()  # turn on interactive mode
        plt.show()

    if verbose:
        print("Mejor distancia: ", mejor_distancia)
        print("Mejor individuo: ", poblacion[0])

    if dibujar_evolucion:
        plt.plot(distancias_iteraciones, label="Distancia mejor individuo")
        plt.plot(distancias_medias, label="Distancia media")
        plt.text(0, 0, "Mejor distancia: {dist}".format(dist=mejor_distancia))
        plt.legend()
        plt.show()
        print(poblacion[0])
        print(distancias_iteraciones[-1])

    return distancias_iteraciones, distancias_medias, poblacion[0]


def dibujar_coordenadas(path: str, cuadricula: bool = True) -> plt:
    """Lee un fichero con coordenadas y las dibuja, retornando el objeto plt para poder seguir dibujando y actualizarlo"""
    coordenadas = leer_coordenadas(path)[0]
    plt.scatter(*zip(*coordenadas))
    if cuadricula:
        plt.grid(color="salmon")
        plt.axvline(0, color="salmon")
        plt.axhline(0, color="salmon")

    return plt


def run():
    ejecutar_ejemplo_viajante_optimizado(
        dibujar_evolucion=False,
        verbose=True,
        parada_en_media=True,
        plot_resultados_parciales=plt,
        parada_en_clones=False,
        elitismo=True,
        cambio_de_mutacion=False,
    )


if __name__ == "__main__":
    num_processes = 10
    processes = []

    for i in range(num_processes):
        process = multiprocessing.Process(target=run)
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    print("All processes have finished")
