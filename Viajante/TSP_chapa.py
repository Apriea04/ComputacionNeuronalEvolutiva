from multiprocessing import Pool
from enums import Seleccion, Mutacion, Crossover, Elitismo
from matplotlib.collections import LineCollection
from Viajante_tradicional_optimizado import (
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
    seleccionar_ruleta_pesos_optimizado,
)
from Datos.datos import leer_coordenadas
import matplotlib.pyplot as plt
import numpy as np

RUTA_COORDENADAS = "Viajante/Datos/50_coordenadas.txt"


def dibujar_individuo(
    individuo: np.ndarray,
    coordenadas: np.ndarray,
    graph_anterior=None,
    sleep_time=0.001,
    distancia: str = "euclidea",
):
    """Dibuja la ruta del mejor individuo y borra la ruta anterior."""
    ruta_x = [coordenadas[i][0] for i in individuo]
    ruta_y = [coordenadas[i][1] for i in individuo]
    ruta_x.append(coordenadas[individuo[0]][0])
    ruta_y.append(coordenadas[individuo[0]][1])
    ruta = np.column_stack((ruta_x, ruta_y))

    match distancia:
        case "euclidea":
            segments = np.array([ruta[:-1], ruta[1:]]).transpose((1, 0, 2))
        case "manhattan":
            segments = np.array([ruta[:-1], ruta[1:]]).transpose((1, 0, 2))
            segments = np.column_stack(
                (segments[:, :, 0], segments[:, :, 1] + segments[:, :, 0])
            )

        case "chebyshev":
            segments = np.array([ruta[:-1], ruta[1:]]).transpose((1, 0, 2))
            segments = np.column_stack(
                (segments[:, :, 0], segments[:, :, 1] + segments[:, :, 0])
            )
            segments = np.column_stack(
                (segments[:, :, 0], segments[:, :, 1] + segments[:, :, 0])
            )

    # Crear un gradiente de color para las líneas
    num_lines = len(individuo) - 1
    color_array = np.linspace(0, 1, num_lines)
    colors = plt.colormaps["brg"](color_array)

    # Crear una colección de líneas con gradiente de color
    lc = LineCollection(segments, colors=colors, linewidth=2)
    plt.gca().add_collection(lc)

    # Borra la ruta anterior si se proporciona
    if graph_anterior is not None:
        graph_anterior.remove()  # Borra el grafo anterior

    plt.pause(sleep_time)  # Pausa para que se vea el cambio

    return lc


# Este método tiene muchos condicionales que no se pueden modificar desde la interfaz gráfica, dado que fueron hechos durante la búsqueda de los mejores parámetros y no han sido eliminados para mostrar que se han probado diferentes configuraciones.
def ejecutar_ejemplo_viajante_optimizado(
    ruta_coordenadas: str,
    dibujar_evolucion: bool = False,
    verbose: bool = True,
    parada_en_media=True,
    max_medias_iguales: int = 10,
    elitismo: bool = True,
    parada_en_clones=False,
    plot_resultados_parciales: bool = True,
    cambio_de_mutacion: bool = False,
    iteraciones: int = 100,
    prob_mutacion: float = 0.13,
    prob_cruzamiento: float = 0.35,
    participantes_torneo: int = 3,
    num_individuos: int = 50,
    tipo_seleccion: Seleccion = Seleccion.TORNEO,
    tipo_mutacion: Mutacion = Mutacion.PERMUTAR_ZONA,
    tipo_crossover: Crossover = Crossover.CROSSOVER_ORDER,
    tipo_elitismo: Elitismo = Elitismo.PASAR_N_PADRES,
    padres_a_pasar_elitismo: int = 3,
):
    global COORDENADAS
    # ----------------------------------------------------------------------
    # Parámetros
    NUM_ITERACIONES = iteraciones  # Comprobar numero Con 10000 iteraciones llega a soluciones muy buenas
    MAX_MEDIAS_IGUALES = max_medias_iguales
    PROB_MUTACION = prob_mutacion  # Visto 0.1
    PROB_CRUZAMIENTO = prob_cruzamiento  # 0.35 puede ser buen numero
    PARTICIPANTES_TORNEO = participantes_torneo
    NUM_INDIVIDUOS = num_individuos
    COORDENADAS, MATRIZ = leer_coordenadas(ruta_coordenadas)
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
        match tipo_seleccion:
            case Seleccion.RULETA_PESOS:
                seleccionados = seleccionar_ruleta_pesos_optimizado(
                    poblacion, aptitud_viajante, MATRIZ, NUM_INDIVIDUOS
                )
            case Seleccion.TORNEO:
                seleccionados = seleccionar_torneo_optimizado(
                    poblacion,
                    PARTICIPANTES_TORNEO,
                    aptitud_viajante,
                    MATRIZ,
                    NUM_INDIVIDUOS,  # TODO: esto depende del crossover
                )

        match tipo_crossover:
            case Crossover.CROSSOVER_PARTIALLY_MAPPED:
                hijos = crossover_partially_mapped_optimizado(
                    seleccionados, aptitud_viajante, MATRIZ, PROB_CRUZAMIENTO
                )
            case Crossover.CROSSOVER_ORDER:
                hijos = crossover_order_optimizado(
                    seleccionados, aptitud_viajante, MATRIZ, PROB_CRUZAMIENTO
                )
            case Crossover.CROSSOVER_CYCLE:
                hijos = crossover_cycle_optimizado(
                    seleccionados, aptitud_viajante, MATRIZ, PROB_CRUZAMIENTO
                )
            case Crossover.EDGE_RECOMBINATION_CROSSOVER:
                hijos = crossover_edge_recombination_optimizado(
                    seleccionados, aptitud_viajante, MATRIZ, PROB_CRUZAMIENTO
                )

            case Crossover.CROSSOVER_PDF:
                hijos = crossover_pdf_optimizado(
                    seleccionados, aptitud_viajante, MATRIZ, PROB_CRUZAMIENTO
                )

        # Mutamos los hijos
        for hijo in hijos:
            if np.random.rand() < PROB_MUTACION:
                match tipo_mutacion:
                    case Mutacion.INTERCAMBIAR_INDICES:
                        hijo = mutar_optimizada(hijo)
                    case Mutacion.PERMUTAR_ZONA:
                        hijo = mutar_desordenado_optimizada(hijo)
                    case Mutacion.INTERCAMBIAR_GENES_VECINOS:
                        hijo = mutar_mejorada_optimizada(hijo)

        # Elitismo
        if elitismo:
            match tipo_elitismo:
                case Elitismo.PADRES_VS_HIJOS:
                    poblacion = elitismo_optimizado(
                    np.concatenate((poblacion, hijos)),
                    NUM_INDIVIDUOS,
                    aptitud_viajante,
                    MATRIZ,
                )
                case Elitismo.PASAR_N_PADRES:
                    poblacion = elitismo_n_padres_optimizado(
                        padres_a_pasar_elitismo,
                        poblacion,
                        hijos,
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

def ejecucion_paralela(
    procesos: int,
    ruta_coordenadas: str,
    dibujar_evolucion: bool = False,
    verbose: bool = True,
    parada_en_media=True,
    max_medias_iguales: int = 10,
    elitismo: bool = True,
    parada_en_clones=False,
    plot_resultados_parciales: bool = True,
    cambio_de_mutacion: bool = False,
    iteraciones: int = 100,
    prob_mutacion: float = 0.13,
    prob_cruzamiento: float = 0.35,
    participantes_torneo: int = 3,
    num_individuos: int = 50,
    tipo_seleccion: Seleccion = Seleccion.TORNEO,
    tipo_mutacion: Mutacion = Mutacion.PERMUTAR_ZONA,
    tipo_crossover: Crossover = Crossover.CROSSOVER_ORDER,
    tipo_elitismo: Elitismo = Elitismo.PASAR_N_PADRES,
    padres_a_pasar_elitismo: int = 3,):
    
    coordenadas, matriz = leer_coordenadas(ruta_coordenadas)
    results = []

    with Pool(processes=procesos) as pool:
        for _ in range(procesos):
            result = pool.apply_async(
                ejecutar_ejemplo_viajante_optimizado,
                args=(
                    ruta_coordenadas,
                    dibujar_evolucion,
                    verbose,
                    parada_en_media,
                    max_medias_iguales,
                    elitismo,
                    parada_en_clones,
                    plot_resultados_parciales,
                    cambio_de_mutacion,
                    iteraciones,
                    prob_mutacion,
                    prob_cruzamiento,
                    participantes_torneo,
                    num_individuos,
                    tipo_seleccion,
                    tipo_mutacion,
                    tipo_crossover,
                    tipo_elitismo,
                    padres_a_pasar_elitismo,
                ),
            )
            results.append(result)

        best_distance = float("inf")
        best_individual = None

        for result in results:
            distances, media, individual = result.get()
            # Imprimimos los resultados:
            print(
                f"Iteraciones: {len(distances)}\tDistancias: {distances[-1]}\tMedia: {media[-1]}"
            )
            if distances[-1] < best_distance:
                best_distance = distances[-1]
                best_individual = individual

    print("\n\n\n\n")
    print("Mejor distance:", best_distance)
    print("Mejor individuo:", best_individual)

    plt.ioff()
    plt.ion()  # turn on interactive mode
    plt.grid(color="salmon")
    plt.axvline(0, color="salmon")
    plt.axhline(0, color="salmon")
    plt.title("Mejor Individuo")
    plt.scatter(*zip(*coordenadas))
    dibujar_individuo(
        best_individual, coordenadas, distancia="euclidea", sleep_time=6000
    )

if __name__ == "__main__":
    ejecucion_paralela(procesos=10, ruta_coordenadas="Viajante/Datos/50_coordenadas.txt", dibujar_evolucion=False, verbose=True, parada_en_media=True, max_medias_iguales=10, elitismo=True, parada_en_clones=False, plot_resultados_parciales=False, cambio_de_mutacion=False, iteraciones=10000, prob_mutacion=0.13, prob_cruzamiento=0.35, participantes_torneo=2, num_individuos=100, tipo_seleccion=Seleccion.TORNEO, tipo_mutacion=Mutacion.PERMUTAR_ZONA, tipo_crossover=Crossover.EDGE_RECOMBINATION_CROSSOVER, tipo_elitismo=Elitismo.PASAR_N_PADRES, padres_a_pasar_elitismo=1)
    