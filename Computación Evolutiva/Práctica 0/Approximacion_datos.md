<a name="br1"></a> 

Grado en Ingeniería Informática

2023/2024

COMPUTACIÓN NEURONAL Y EVOLUTIVA: PRÁCTICAS

Prácitca 1 (parte 2): aproximación de datos mediante una neurona artiﬁcial analógica

APELLIDOS, NOMBRE:

(mayúsculas)

1\. Enunciado

El objetivo de esta práctica es entrenar una neurona artiﬁcial para que sea capaz de detectar si un paciente puede

o menos sufrir de apendicitis.

En primer lugar hay que descargar los archivos appendicitis.dat y muestra\_pacientes.dat de la base de datos keel

Dataset ([enlace](https://sci2s.ugr.es/keel/dataset.php?cod=183))

Una vez la neurona pueda leer el archivo se pide

1\. Añadir como opción una función de salida que proceda de f (p) = sin(p) en [−1, 1] con imagen en el inter-

valo apropiado.

2\. Añadir como opción una función de salida que proceda de f (p) = 1/(1+ ex p(−p)) en (0, 1), con imagen en

el intervalo apropiado

3\. Añadir como opción una función de salida que proceda de f (p) = ex p(−p<sup>2</sup>) en (0, 1] con imagen en el

intervalo apropiado

4\. Añadir como opción una función de salida que proceda de f (p) = p/(1 + (p<sup>2</sup>)) en [−0,5, 0,5], con imagen

en el intervalo apropiado

5\. hallar los puntos de inﬂexión de las funciones anteriores

6\. Añadir la opción de enfriamiento simulado al la razón de aprendizaje. La razón η ha de ser una función

decreciente que dependa del tiempo. Por ejemplo η(t) = −1/(1 + ex p(−4m(t − c))) + 1 siendo m y c

parámetros a ajustar

7\. Ajustar los valores de los parámetros error aceptable, ratio de aprendizaje y tiempo máximo para obtener

un mínimo de 98 muestras bien aproximadas. Escribir las pruebas con las que se ha obtenido la mejor

aproximación. Función de salida, t<sub>max</sub>, η(m, c), error normalizado/muestras

8\. utilizar la neurona con la mejor opción para diagnosticar al paciente

paciente = (0,098, 0,607, 0,123, 0,042, 0,016, 0,67, 0,105),

(1.1)

si tiene apendicitis (Si/No/Varia) mejor opción: función de salida t<sub>max</sub>, η(m, c) error normalizado/muestras

Dpto. de Matemáticas. Universidad de León.

Página 1 de [1](#br1)

