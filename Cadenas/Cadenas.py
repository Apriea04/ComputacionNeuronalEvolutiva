from Genética_1000_unos import mutar, crossover, aptitud, seleccionar_ruleta
import numpy as np

INDIVIDUOS = 100
GENES = 100
MUTACION = 0.1

#Crear una poblacion de 12 individuos con 100 genes conteniendo 0s y 1s
poblacion = np.random.choice([0,1], size=(INDIVIDUOS, GENES))

mejores_aptitudes = []
aptitud_promedio = []
mejor_aptitud_actual = 0

max_iteraciones = 5000

while mejor_aptitud_actual < 100 and len(mejores_aptitudes) < max_iteraciones:
    nueva_generacion = []
    while len(nueva_generacion) < INDIVIDUOS:
        padres = seleccionar_ruleta(poblacion, 2)
        nuevo_individuo = crossover(padres)
        nueva_generacion.append(nuevo_individuo)
    nueva_generacion = np.array(nueva_generacion)
    mutar(nueva_generacion, MUTACION)
    aptitudes = [aptitud(individuo) for individuo in nueva_generacion]
    mejor_aptitud_actual = np.max(aptitudes)
    mejores_aptitudes.append(mejor_aptitud_actual)
    aptitud_promedio.append(np.mean(aptitudes))
    poblacion = nueva_generacion
    
    if len(mejores_aptitudes) % 100 == 0:
        print(f'Generación {len(mejores_aptitudes)} Mejor aptitud {mejor_aptitud_actual}, aptitud promedio {aptitud_promedio[-1]}')

print(mejores_aptitudes[-1])

# Usar matplotlib para graficar la aptitud promedio y la mejor aptitud
import matplotlib.pyplot as plt
plt.plot(aptitud_promedio, label='Aptitud promedio')
plt.plot(mejores_aptitudes, label='Mejor aptitud')
plt.xlabel('Generación')
plt.ylabel('Aptitud')
plt.legend()
plt.show()