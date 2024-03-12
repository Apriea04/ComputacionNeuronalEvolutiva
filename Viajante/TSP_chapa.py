from Viajante_tradicional_optimizado import (
    leer_distancias_optimizada,
    crear_poblacion_optimizada,
    aptitud_viajante,
    seleccionar_torneo_optimizado,
    crossover_partially_mapped_optimizado,
    crossover_edge_recombination_optimizado,
    mutar_optimizada,
    elitismo_optimizado,
)
from Datos.datos import leer_coordenadas
import matplotlib.pyplot as plt
import numpy as np

def dibujar_individuo(individuo: np.ndarray, coordenadas: np.ndarray, graph_anterior=None, sleep_time=0.001):
    """Dibuja la ruta del mejor individuo y borra la ruta anterior."""
    ruta_x = [coordenadas[i][0] for i in individuo]
    ruta_y = [coordenadas[i][1] for i in individuo]
    graph = plt.plot(ruta_x + [ruta_x[0]], ruta_y + [ruta_y[0]], marker='o', linestyle='-', color='blue')[0]

    # Borra la ruta anterior si se proporciona
    if graph_anterior is not None:
        graph_anterior.remove()  # Borra el grafo anterior
        
    plt.pause(sleep_time)  # Pausa para que se vea el cambio
    
    return graph

# ----------------------------------------------------------------------
# Parámetros
NUM_ITERACIONES = 10000
MAX_MEDIAS_IGUALES = 30
PROB_MUTACION = 0.1
PROB_CRUZAMIENTO = 0.8
PARTICIPANTES_TORNEO = 3
NUM_INDIVIDUOS = 100
RUTA_MATRIZ = "Viajante/Datos/10_distancias.txt"
RUTA_COORDENADAS = "Viajante/Datos/10_coordenadas.txt"
MATRIZ = leer_distancias_optimizada(RUTA_MATRIZ)
COORDENADAS = leer_coordenadas(RUTA_COORDENADAS)
# ----------------------------------------------------------------------


def ejecutar_ejemplo_viajante_optimizado(
    dibujar_evolucion: bool = False,
    verbose: bool = True,
    parada_en_media=False,
    parada_en_clones=False,
    plot_resultados_parciales: bool = True,
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
            
    if plot_resultados_parciales:
        plt.ion() # turn on interactive mode
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
            NUM_INDIVIDUOS * 2,
        )

        # Cruzamos los seleccionados
        hijos = crossover_edge_recombination_optimizado(
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
        
        if plot_resultados_parciales:
            grafico_nuevo = dibujar_individuo(poblacion[0], COORDENADAS, grafico_nuevo, sleep_time=0.25)
        
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

        if (
            parada_en_media
            and i > MAX_MEDIAS_IGUALES
            and distancias_medias[-1] == distancias_medias[-10]
        ):
            break
        
        if parada_en_clones:
            if len(set([aptitud_viajante(individuo, MATRIZ) for individuo in poblacion])) == 1:
                break

    # Ordena la población según la aptitud
    indices_ordenados = np.argsort([aptitud_viajante(sol, MATRIZ) for sol in poblacion])
    # Reordena la población
    poblacion = poblacion[indices_ordenados]

    mejor_distancia = aptitud_viajante(poblacion[0], MATRIZ)
    
    if plot_resultados_parciales:
        plt.ioff() # turn on interactive mode
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
    coordenadas = leer_coordenadas(path)
    plt.scatter(*zip(*coordenadas))
    if cuadricula:
        plt.grid(color="salmon")
        plt.axvline(0, color="salmon")
        plt.axhline(0, color="salmon")

    return plt


if __name__ == "__main__":
    dibujar_coordenadas("Viajante/Datos/10_coordenadas.txt")
    ejecutar_ejemplo_viajante_optimizado(
        dibujar_evolucion=True, verbose=True, parada_en_media=True, plot_resultados_parciales=plt, parada_en_clones=True
    )