import random
from deap import base, creator, tools, algorithms
import numpy as np
import matplotlib.pyplot as plt
from Datos.datos import leer_coordenadas
from Viajante_tradicional_optimizado import leer_distancias_optimizada, aptitud_viajante
from TSP_chapa import dibujar_individuo
import concurrent.futures
from enums import Elitismo, Seleccion, Crossover, Mutacion
 # ----------------------------------------------------------------------
# Parámetros
NUM_EJECUCIONES = 10
NUM_ITERACIONES = (
    20000  # Comprobar numero Con 10000 iteraciones llega a soluciones muy buenas
)
PROB_MUTACION = 0.13  # Visto 0.13
PROB_CRUZAMIENTO = 0.35  # 0.35 puede ser buen numero
PARTICIPANTES_TORNEO = 3
NUM_INDIVIDUOS = 100
RUTA_COORDENADAS = "Viajante/Datos/50_coordenadas.txt"
COORDENADAS, MATRIZ = leer_coordenadas(RUTA_COORDENADAS)
CROSSOVER = Crossover.CROSSOVER_ORDER
SELECCION = Seleccion.TORNEO
MUTACION = Mutacion.PERMUTAR_ZONA
ELITISMO = Elitismo.PASAR_N_PADRES
PADRES_A_PASAR_ELITISMO = 3

# ----------------------------------------------------------------------


# Adaptar la función de aptitud para que reciba un individuo y devuelva su aptitud
def fitness (individuo):
    return (aptitud_viajante(individuo, MATRIZ),)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(len(COORDENADAS)), len(COORDENADAS))
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=NUM_INDIVIDUOS)

match (CROSSOVER):
    case Crossover.CROSSOVER_ORDER:
        toolbox.register("mate", tools.cxOrdered)
    case Crossover.CROSSOVER_PARTIALLY_MAPPED:
        toolbox.register("mate", tools.cxPartialyMatched)
    case _:
        raise ValueError("Crossover no implementado")

match (MUTACION):
    case Mutacion.PERMUTAR_ZONA:
        # No permuta, pero invierte el orden entre dos índices. Es lo más parecido que incluye DEAP
        toolbox.register("mutate", tools.mutInversion)
    case Mutacion.INTERCAMBIAR_INDICES:
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)
    case _:
        raise ValueError("Mutación no implementada")
        
match (SELECCION):
    case Seleccion.TORNEO:
        toolbox.register("select", tools.selTournament, tournsize=PARTICIPANTES_TORNEO)
    case Seleccion.RULETA:
        toolbox.register("select", tools.selRoulette)
    case _:
        raise ValueError("Selección no implementada")

toolbox.register("evaluate", fitness)

def main():
    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    match (ELITISMO):
        case Elitismo.PADRES_VS_HIJOS:
            pop, log = algorithms.eaSimple(pop, toolbox, cxpb=PROB_CRUZAMIENTO, mutpb=PROB_MUTACION, ngen=NUM_ITERACIONES, stats=stats, halloffame=hof, verbose=True)
        case Elitismo.PASAR_N_PADRES:
            pop, log = algorithms.eaMuPlusLambda(pop, toolbox, cxpb=PROB_CRUZAMIENTO, mutpb=PROB_MUTACION, ngen=NUM_ITERACIONES, lambda_=NUM_INDIVIDUOS-PADRES_A_PASAR_ELITISMO, mu=PADRES_A_PASAR_ELITISMO, stats=stats, halloffame=hof, verbose=True)

    return pop, stats, hof, log

def evaluate_solution(best_solution, best_distance):
    pop, stats, hof, log = main()

    if hof[0].fitness.values[0] < best_distance:
        return hof[0], hof[0].fitness.values[0]
    else:
        return best_solution, best_distance

if __name__ == "__main__":
    best_solution = None
    best_distance = float('inf')
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        
        for _ in range(NUM_EJECUCIONES):
            future = executor.submit(evaluate_solution, best_solution, best_distance)
            futures.append(future)
        
        for future in concurrent.futures.as_completed(futures):
            best_solution, best_distance = future.result()
    
    # Imprimimos la mejor solución encontrada
    print("Mejor individuo: ", best_solution)
    print("Mejor distancia: ", best_distance)
    
    plt.ioff()
    plt.ion()  # turn on interactive mode
    plt.grid(color="salmon")
    plt.axvline(0, color="salmon")
    plt.axhline(0, color="salmon")
    plt.title("Mejor Individuo")
    plt.scatter(*zip(*COORDENADAS))
    dibujar_individuo(best_solution, COORDENADAS, distancia="euclidea", sleep_time=6000)