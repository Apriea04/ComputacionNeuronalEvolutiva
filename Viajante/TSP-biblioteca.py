from multiprocessing import Pool
import random
from deap import base, creator, tools, algorithms
import numpy as np
import matplotlib.pyplot as plt
from Datos.datos import leer_coordenadas
from Viajante_tradicional_optimizado import leer_distancias_optimizada, aptitud_viajante
from TSP_chapa import dibujar_individuo
import concurrent.futures
from enums import Elitismo, Seleccion, Crossover, Mutacion

# Adaptar la función de aptitud para que reciba un individuo y devuelva su aptitud
def fitness (individuo):
    return (aptitud_viajante(individuo, MATRIZ),)

def ejecutar(ruta_coordenadas: str, verbose: bool, iteraciones: int, prob_mutacion: float, prob_cruzamiento: float, participantes_torneo: int, num_individuos: int, tipo_seleccion: Seleccion, tipo_mutacion: Mutacion, tipo_crossover: Crossover, tipo_elitismo: Elitismo, padres_a_pasar_elitismo: int, best_solution, best_distance):
    
    pop, stats, hof, log = ejecutar_ejemplo_viajante(ruta_coordenadas, verbose, iteraciones, prob_mutacion, prob_cruzamiento, participantes_torneo, num_individuos, tipo_seleccion, tipo_mutacion, tipo_crossover, tipo_elitismo, padres_a_pasar_elitismo)

    if hof[0].fitness.values[0] < best_distance:
        return hof[0], hof[0].fitness.values[0]
    else:
        return best_solution, best_distance
    
def ejecutar_ejemplo_viajante(ruta_coordenadas: str, verbose: bool, iteraciones: int, prob_mutacion: float, prob_cruzamiento: float, participantes_torneo: int, num_individuos: int, tipo_seleccion: Seleccion, tipo_mutacion: Mutacion, tipo_crossover: Crossover, tipo_elitismo: Elitismo, padres_a_pasar_elitismo: int):
    global MATRIZ
    coordenadas, MATRIZ = leer_coordenadas(ruta_coordenadas)
    
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, range(len(coordenadas)), len(coordenadas))
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=num_individuos)

    match (tipo_crossover):
        case Crossover.CROSSOVER_ORDER:
            toolbox.register("mate", tools.cxOrdered)
        case Crossover.CROSSOVER_PARTIALLY_MAPPED:
            toolbox.register("mate", tools.cxPartialyMatched)
        case _:
            raise ValueError("Crossover no implementado")

    match (tipo_mutacion):
        case Mutacion.PERMUTAR_ZONA:
            # No permuta, pero invierte el orden entre dos índices. Es lo más parecido que incluye DEAP
            toolbox.register("mutate", tools.mutInversion)
        case Mutacion.INTERCAMBIAR_INDICES:
            toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)
        case _:
            raise ValueError("Mutación no implementada")
            
    match (tipo_seleccion):
        case Seleccion.TORNEO:
            toolbox.register("select", tools.selTournament, tournsize=participantes_torneo)
        case Seleccion.RULETA:
            toolbox.register("select", tools.selRoulette)
        case _:
            raise ValueError("Selección no implementada")

    toolbox.register("evaluate", fitness)
    
    pop = toolbox.population(n=num_individuos)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    match (tipo_elitismo):
        case Elitismo.PADRES_VS_HIJOS:
            pop, log = algorithms.eaSimple(pop, toolbox, cxpb=prob_cruzamiento, mutpb=prob_mutacion, ngen=iteraciones, stats=stats, halloffame=hof, verbose=verbose)
        case Elitismo.PASAR_N_PADRES:
            pop, log = algorithms.eaMuPlusLambda(pop, toolbox, cxpb=prob_cruzamiento, mutpb=prob_mutacion, ngen=iteraciones, lambda_=num_individuos-padres_a_pasar_elitismo, mu=num_individuos, stats=stats, halloffame=hof, verbose=verbose)

    return pop, stats, hof, log
    
def ejecucion_paralela(procesos, ruta_coordenadas: str, verbose: bool, iteraciones: int, prob_mutacion: float, prob_cruzamiento: float, participantes_torneo: int, num_individuos: int, tipo_seleccion: Seleccion, tipo_mutacion: Mutacion, tipo_crossover: Crossover, tipo_elitismo: Elitismo, padres_a_pasar_elitismo: int):
    best_solution = None
    best_distance = float('inf')
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        
        for _ in range(procesos):
            future = executor.submit(ejecutar, ruta_coordenadas, verbose, iteraciones, prob_mutacion, prob_cruzamiento, participantes_torneo, num_individuos, tipo_seleccion, tipo_mutacion, tipo_crossover, tipo_elitismo, padres_a_pasar_elitismo,best_solution, best_distance)
            futures.append(future)
        
        for future in concurrent.futures.as_completed(futures):
            best_solution, best_distance = future.result()
    
    # Imprimimos la mejor solución encontrada
    print("Mejor individuo: ", best_solution)
    print("Mejor distancia: ", best_distance)
    
    coordenadas = leer_coordenadas(ruta_coordenadas)[0]
    
    plt.ioff()
    plt.ion()  # turn on interactive mode
    plt.grid(color="salmon")
    plt.axvline(0, color="salmon")
    plt.axhline(0, color="salmon")
    plt.title("Mejor Individuo")
    plt.scatter(*zip(*coordenadas))
    dibujar_individuo(best_solution, coordenadas, distancia="euclidea", sleep_time=6000)

if __name__ == "__main__":
    ejecucion_paralela(3, "Viajante/Datos/50_coordenadas.txt", True, 1000, 0.13, 0.35, 2, 100, Seleccion.TORNEO, Mutacion.PERMUTAR_ZONA, Crossover.CROSSOVER_ORDER, Elitismo.PASAR_N_PADRES, 3)