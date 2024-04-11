# TSP Algoritmos genéticos
## Descripción
Práctica de Algoritmos Genéticos con el TSP (Travelling Salesman Problem) en Python.
Se ha orientado a la aplicación práctica de un robot taladrador que debe taladrar una serie de agujeros en una placa metálica.
## Ejecución
Se recomienda utilizar Python 3.11.8 en un entorno virtual.

Para ejecutar el código, es necesario instalar las dependencias que se encuentran en el fichero requirements.txt. Para ello, se puede ejecutar el siguiente comando:
```bash
pip install -r requirements.txt
```
Una vez instaladas las dependencias, se puede ejecutar el código con el siguiente comando:
```bash
python gui.py
```

**Para salir es necesario cerrar la terminal desde donde se lanzó el programa.**

Para las coordenadas, en el directorio /Datos se ajuntan algunos ejemplos.

Se deben ignorar los posibles warnings y errores que se muestren en la consola, dado que son fruto de una interfaz gráfica frágil con el mero propósito de simplificar el ajuste de parámetros.
## Contenido relevante
- **gui.py**: Interfaz gráfica que permite el ajuste y la ejecución del código. Internamente ejecuta TSP_biblioteca.py y TSP_chapa.py.
- **TSP_chapa.py**: Código desarrollado para la resolución del problema sin uso de bibliotecas.
- **TSP_biblioteca.py**: Código desarrollado en primera instancia para la resolución del problema con uso de la biblioteca DEAP.
- **Viajante_tradicional.py**: Código desarrollado para la resolución del problema sin el uso de bibliotecas. Al ejecutarlo, se muestra cómo la media se estanca. El problema no se corrigió en ese fichero.
- **Viajante_tradicional_optimizado.py**: Código desarrollado para la resolución del problema con el uso de NumPy. Se convirtieron con el uso de IA los métodos de Viajante_tradicional.py y agregaron nuevos métodos para otros operadores. Al ejecutarlo, el problema del estancamiento de la media persiste. Sin embargo, los métodos son usados en TSP_chapa.py y TSP_biblioteca.py.
- **requirements.txt**: Fichero que contiene las dependencias necesarias para la ejecución del código.
- **/Datos**: Carpeta que contiene los datos de las coordenadas de los agujeros a taladrar. Se adjunta también código para poder generar nuevas coordenadas.