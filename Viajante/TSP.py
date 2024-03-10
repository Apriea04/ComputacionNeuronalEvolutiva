import numpy as np
from python_tsp.exact import solve_tsp_dynamic_programming, solve_tsp_brute_force
from Viajante_tradicional import leer_distancias

_, distances = leer_distancias(
    "Viajante/Datos/matriz6.data", "Viajante/Datos/pueblos6.txt"
)

distance_matrix = np.array(distances)
import sys

sys.setrecursionlimit(2000)  # Aumenta el límite de recursión a 3000
permutation, distance = solve_tsp_dynamic_programming(distance_matrix)
# A distancia restamos la distancia de vuelta al punto de partida
print(permutation, distance)