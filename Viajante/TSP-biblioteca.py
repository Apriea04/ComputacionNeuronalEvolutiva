import random
from deap import base, creator, tools, algorithms
import numpy as np
import matplotlib.pyplot as plt
from Datos.datos import leer_coordenadas
from Viajante_tradicional_optimizado import leer_distancias_optimizada, aptitud_viajante
from TSP_chapa import dibujar_individuo
import concurrent.futures
 # ----------------------------------------------------------------------
# Parámetros
NUM_EJECUCIONES = 4
NUM_ITERACIONES = (
    10000  # Comprobar numero Con 10000 iteraciones llega a soluciones muy buenas
)
PROB_MUTACION = 0.13  # Visto 0.13
PROB_CRUZAMIENTO = 0.35  # 0.35 puede ser buen numero
PARTICIPANTES_TORNEO = 3
NUM_INDIVIDUOS = 50
RUTA_COORDENADAS = "Viajante/Datos/50_coordenadas.txt"
COORDENADAS, MATRIZ = leer_coordenadas(RUTA_COORDENADAS)
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

toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=PARTICIPANTES_TORNEO)
toolbox.register("evaluate", fitness)

def main():
    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=PROB_CRUZAMIENTO, mutpb=PROB_MUTACION, ngen=NUM_ITERACIONES, stats=stats, halloffame=hof, verbose=True)

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