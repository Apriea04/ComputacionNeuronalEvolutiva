def leer_distancias(path: str) -> tuple:
    """Lee una matriz distancias de un fichero de texto.
    :param path: Ruta del fichero.
    :return: lista de nombres, matriz de distancias.
    """
    # Inicializamos una lista para los nombres de los municipios y una lista de listas para la matriz de números
    nombres_municipios = []
    matriz_numeros = []

    # Abrimos el archivo para lectura
    with open(path, 'r', encoding='utf-8') as archivo:
        # Iteramos sobre cada línea del archivo
        for linea in archivo:
            # Dividimos la línea en dos partes: el nombre del municipio y los números
            # Primero eliminamos las comillas y luego dividimos usando el primer espacio encontrado
            partes = linea.strip().split('"')
            nombre_municipio = partes[1].strip()
            numeros = partes[2].strip().split()

            # Añadimos el nombre del municipio a la lista de nombres
            nombres_municipios.append(nombre_municipio)
            
            # Convertimos los números a enteros y los añadimos a la matriz
            matriz_numeros.append([int(numero) for numero in numeros])

    # Devolvemos los nombres y la matriz
    return nombres_municipios, matriz_numeros


#Probamos la funcion anterior
nombres, matriz = leer_distancias('Viajante/Distancias_ejemplo.txt')
print(nombres)
print(matriz)