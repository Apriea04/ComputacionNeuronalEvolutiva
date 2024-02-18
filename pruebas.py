from Genética_1000_unos import mutar, crossover, aptitud
import numpy as np


    
# ----------------------------------------------------------------
#crear array de 2x10 con 0s y 1s
array_2x10 = np.random.randint(2, size=(2, 10))

sucesor = crossover(array_2x10, verbose=True)
array_2x10 = np.vstack((array_2x10, sucesor))
print(array_2x10)

aptitudes = np.array([aptitud(individuo) for individuo in array_2x10])