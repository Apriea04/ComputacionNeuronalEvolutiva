from Genética_1000_unos import mutar, crossover, aptitud, seleccionar_padres
import numpy as np

#Crear una poblacion de 12 individuos con 100 genes conteniendo 0s y 1s
poblacion = np.random.choice([0,1], size=(12, 100))

mejores_aptitudes = []
aptitud_promedio = []
mejor_aptitud_actual = 0

while mejor_aptitud_actual < 100:
    nueva_generacion = []
    while len(nueva_generacion) < 12:
        padres = seleccionar_padres(poblacion, 2)
        nuevo_individuo = crossover(padres)
        nueva_generacion.append(nuevo_individuo)
    nueva_generacion = np.array(nueva_generacion)
    mutar(nueva_generacion, 0.01)
    aptitudes = [aptitud(individuo) for individuo in nueva_generacion]
    mejor_aptitud_actual = np.max(aptitudes)
    mejores_aptitudes.append(mejor_aptitud_actual)
    aptitud_promedio.append(np.mean(aptitudes))
    poblacion = nueva_generacion

print(mejores_aptitudes[-1])

# Usar matplotlib para graficar la aptitud promedio y la mejor aptitud
import matplotlib.pyplot as plt
plt.plot(aptitud_promedio, label='Aptitud promedio')
plt.plot(mejores_aptitudes, label='Mejor aptitud')
plt.xlabel('Generación')
plt.ylabel('Aptitud')
plt.legend()
plt.show()