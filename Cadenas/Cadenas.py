import random
import matplotlib.pyplot as plt

N = 10

def seleccion_ruleta(poblacion, n):
    """Selecciona n individuos de la población usando el método de la ruleta"""
    aptitudes = [aptitud(individuo) for individuo in poblacion]
    poblacion_seleccionada = []
    aptitudes_acumuladas = [sum(aptitudes[: i + 1]) for i in range(len(aptitudes))]
    aptitud_total = aptitudes_acumuladas[-1]
    for _ in range(n):
        valor = random.randint(0, aptitud_total)
        
        seleccionado = 0
        
        while not aptitudes_acumuladas[seleccionado] >= valor:
                seleccionado += 1
        poblacion_seleccionada.append(poblacion[seleccionado])
    return poblacion_seleccionada

def crossover_simple(progenitores):
    """Realiza el crossover simple entre dos individuos. Devuelve el hijo resultante."""
    indice_corte = random.randint(0, len(progenitores[0]) - 1)
    return progenitores[0][:indice_corte] + progenitores[1][indice_corte:]

def crear_poblacion_cadenas(n, longitud):
    """Crea una población de n individuos, cada uno de longitud longitud."""
    poblacion = []
    for i in range(n):
        poblacion.append([])
        for _ in range(longitud):
            poblacion[i].append(random.choice([0, 1]))
    return poblacion

def aptitud(individuo):
    """Calcula la aptitud de un individuo, que es la suma de sus valores."""
    return sum([int(x) for x in individuo])

def mutar(individuo):
    """Muta un individuo, cambiando uno de sus valores aleatoriamente."""
    indice = random.randint(0, len(individuo) - 1)
    nuevo_valor = 1 if individuo[indice] == 0 else 0
    individuo[indice] = nuevo_valor
    return individuo

poblacion = crear_poblacion_cadenas(N, 100)
mejores_aptitudes = []
aptitud_promedio = []
for _ in range(1000):
    aptitud_promedio.append(sum([aptitud(individuo) for individuo in poblacion]) / N)
    mejores_aptitudes.append(max([aptitud(individuo) for individuo in poblacion]))
    #Creamos una nueva generación
    nueva_generacion = []
    for _ in range(N):
        padres = seleccion_ruleta(poblacion, 2)
        hijo = crossover_simple(padres)
        #Mutamos el hijo con una probabilidad del 1%
        if random.random() < 0.01:
            hijo = mutar(hijo)
        nueva_generacion.append(hijo)
    poblacion = nueva_generacion

#Dibujamos la gráfica del funcionamiento
plt.plot(aptitud_promedio, label='Aptitud promedio')
plt.plot(mejores_aptitudes, label='Mejor aptitud')
plt.xlabel('Generación')
plt.ylabel('Aptitud')
plt.legend()
plt.show()