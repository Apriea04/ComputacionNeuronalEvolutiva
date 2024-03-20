import random
from deap import base, creator, tools, algorithms
import numpy as np
import matplotlib.pyplot as plt
from Datos.datos import leer_coordenadas
from Viajante_tradicional_optimizado import leer_distancias_optimizada, aptitud_viajante
from TSP_chapa import dibujar_individuo

RUTA_COORDENADAS = "Viajante/Datos/50_coordenadas.txt"
COORDENADAS, MATRIZ = leer_coordenadas(RUTA_COORDENADAS)

# Adaptar la función de aptitud para que reciba un individuo y devuelva su aptitud
def fitness (individuo):
    return (aptitud_viajante(individuo, MATRIZ),)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(50), 50)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", fitness)

def main():
    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.35, mutpb=0.13, ngen=5000, stats=stats, halloffame=hof, verbose=True)

    return pop, stats, hof, log

if __name__ == "__main__":
    pop, stats, hof, log = main()
    
    # Imprimimos la menor distancia encontrada
    print("Mejor individuo: ", hof[0])
    print("Mejor distancia: ", hof[0].fitness.values[0])
    
    plt.ioff()
    plt.ion()  # turn on interactive mode
    plt.grid(color="salmon")
    plt.axvline(0, color="salmon")
    plt.axhline(0, color="salmon")
    plt.title("Mejor Individuo")
    plt.scatter(*zip(*COORDENADAS))
    dibujar_individuo(hof[0], COORDENADAS, distancia="euclidea", sleep_time=20)
    