import Viajante_tradicional_optimizado as TSP
from enum import Enum


# Clases con enumeraciones de los tipos de selección, mutación y crossover
class Seleccion(Enum):
    RULETA_PESOS = 1
    TORNEO = 2

class Mutacion(Enum):
    INTERCAMBIAR_INDICES = 1
    PERMUTAR_ZONA = 2
    INTERCAMBIAR_GENES_VECINOS = 3
    
class Crossover(Enum):
    CROSSOVER_PARTIALLY_MAPPED = 1
    CROSSOVER_ORDER = 2
    CROSSOVER_CYCLE = 3
    EDGE_RECOMBINATION_CROSSOVER = 4
    CROSSOVER_PDF = 5
    
class Elitismo(Enum):
    PADRES_VS_HIJOS = 1
    PASAR_N_PADRES = 2
    
# Hacer un mapeo entre las enumeraciones y las funciones de Viajante_tradicional_optimizado.py
class EnumMapper:
    @staticmethod
    def seleccion(seleccion: Seleccion) -> callable:
        if seleccion == Seleccion.RULETA_PESOS:
            return TSP.seleccionar_ruleta_pesos_optimizado
        elif seleccion == Seleccion.TORNEO:
            return TSP.seleccionar_torneo_optimizado
        else:
            return None

    @staticmethod
    def mutacion(mutacion: Mutacion) -> callable: 
        if mutacion == Mutacion.INTERCAMBIAR_INDICES:
            return TSP.mutar_optimizada
        elif mutacion == Mutacion.PERMUTAR_ZONA:
            return TSP.mutar_desordenado_optimizada
        elif mutacion == Mutacion.INTERCAMBIAR_GENES_VECINOS:
            return TSP.mutar_mejorada_optimizada
        else:
            return None

    @staticmethod
    def crossover(crossover: Crossover) -> callable:
        if crossover == Crossover.CROSSOVER_PARTIALLY_MAPPED:
            return TSP.crossover_partially_mapped_optimizado
        elif crossover == Crossover.CROSSOVER_ORDER:
            return TSP.crossover_order_optimizado
        elif crossover == Crossover.CROSSOVER_CYCLE:
            return TSP.crossover_cycle_optimizado
        elif crossover == Crossover.EDGE_RECOMBINATION_CROSSOVER:
            return TSP.crossover_edge_recombination_optimizado
        elif crossover == Crossover.CROSSOVER_PDF:
            return TSP.crossover_pdf_optimizado
        else:
            return None
    