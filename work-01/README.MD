# Nombre
Vladimir Villegas Luis.  

# Carrera
Ingeniería Informática, 7º Semestre.  

# Materia
Machine Learning.

# Código
Este código se basa en una ecuación gaussiana de densidad de probabilidad (PDF) con dos parámetros basados en el "entrenamiento" con dós imágenes para cada evento (1).

<img src="https://render.githubusercontent.com/render/math?math=P(A_{ij_{objetivo}}| \sigma_{entrenamiento},\mu_{entrenamiento})=\frac{1}{\sqrt{2 \pi \sigma ^2}}e^{\frac{(x-\mu)^2}{2\sigma^2}}">

Esto es la probabilidad de que el pixel en la imagen objetivo A(i,j) sea arena o mar.

El modelo de clasificación propuesto con los parámetros sigma y mu de cada set de entrenamiento, se puede encontrar en la siguiente imagen.

![alt text](https://github.com/devnull404/VV-TECNM/blob/master/work-01/img/modelo.jpg)

![Markdown Mark](/img/modelo.jpg)


# Importando módulos necesarios
En primera instancia, debemos instalar los módulos necesarios para poder trabajar con imágnes, una de estas es NUMPY, y MatplotLib, que nos permite graficar, entre muchas otras cosas, utilizando el código siguiente:  

```python
import matplotlib.pyplot as plt
import numpy as np
import json
```

# Guardando parámetros para multiples runs del código
Un módulo utilizado para aligerar la carga en el CPU para realizar pruebas muchas veces, es el JSON, que nos permite trabajar con formatos .json para guardar los datos parámetros que calculemos en el disco duro como un archivo denominado data.json y no realizar los mismos cálculos una y otra vez. Entonces, con las líneas de código siguientes podemos cargar al programa los datos contenidos en este archivo para llamarlos incluso después de que se haya ejecutado el programa:

```python
with open('data.json', 'r+') as fp:
    data = json.load(fp)
```

# Leyendo las imágenes
Después de haber cargado todas las librerías necesarias, podemos empezar cargando las imágenes para empezar a trabajar con ellas utilizando la librería **MatplotLib**, que llamamos en la primer sección como **plt**, entonces, cargamos en tres variables cada imagen con la función **plt.imread()**:

```python
sea, sand, sea_sand = [plt.imread("./img/sea.jpg"), plt.imread("./img/sand.jpg"), plt.imread("./img/sea_sand.jpg")]
```

# Creando función para transponer una imagen dada
Como sería ideal que todas las imágenes tuvieran el mismo tamaño y orientación, en este caso una se entuentra rotada, por lo que es necesario realizar una transformación, que en este caso, funcionaría una transpuesta para pasar de **(m,n)** => **(n,m)**, que podría realizarse con la librería **NUMPY**, pero haremos de forma manual para entender bien el procedimiento. El código que se muestra a continuación crea una imagen auxiliar vacía con dimensiones invertidas **np.empty(b, a, c)** obteniendo los valores con el método **.shape**, que nos da las dimensiones de la imagen de entrada en la función. Finalmente, se pasa por cada pixel con dos ciclos *For* anidados para pasar por cada fila y collumna, para intercambiar las comumnas por las filas mediante **aux[j][i] = img[i][j]**

```python
def transpuesta(img):
    a, b, c = img.shape
    aux = np.empty((b, a, c), dtype=int)
    for i in range(a):
        for j in range(b):
            aux[j][i] = img[i][j]
    return aux
```
# Función para calcular el mean (µ) de cada imagen
Podemos realizarlo de manera manual, calculando por fórmula, pasando por cada pixel en cada imagen y encontrar sus means, por lo que de la misma manera que en el segmento de código anterior, utilizamos un ciclo **For** anidado, pasando por cada pixel y guardando la suma de cada uno en un contador, que al final se agrega a la fórmula para regresar el resultado.

```python
def mu(img):
    a, b, c = img.shape
    count = [0, 0, 0]
    for i in range(a):
        for j in range(b):
            count[0] += img[i][j][0]
            count[1] += img[i][j][1]
            count[2] += img[i][j][2]
    count[0], count[1], count[2] = [count[0]/(a*b), count[1]/(a*b), count[2]/(a*b)]
    return count
```

# Función para calcular Sigma (σ) 
Por otra parte, generamos otra función que calcula el σ de cada imagen, y tiene como datos de entrada los valores µ calculados de cada imagen, por lo que sólo es necesario llamar a esta para obtener los dos valores (µ y σ).

```python
def sigma(img):
    mm = mu(img)
    m, n, c = img.shape
    count = 0
    suma = 0
    for i in range(m):
        for j in range(n):
            count += (img[i][j]-mm)**2
    print(np.sqrt(count/(n*m)))
    return list(np.sqrt(count/(n*m))), mm
```

# Función para calcular la PDF
Esta función simplemente toma los parámetros estadísticos de cada imagen, y los calcula para regresar el valor de la PDF.

```python
def pdf(m, o, pixel):
    return (1/np.sqrt(2*np.pi*(o**2)))*((np.e)**(-((pixel-m)**2)/(2*(o**2))))
```

## Primer run y guardado de los parámetros estadísticos en unarchivo JSON

Ahora, corremos las funciones **mu** y **sigma** para obtener sus parámetros y guardarlos en disco y no tener que llamar a la función **mu** y **sigma** ya que son pesadas para el CPU, ya que cada imagen es de un tamaño de 1200X1600, lo que da como resultado casi 2e^7 pixeles, sobre los cuales se tiene que pasan las funciones.

```python
    aux1, aux2 = sigma(sea)
    data["sea"] = {"means": aux2, "sigmas": aux1}
    aux1, aux2 = sigma(sand)
    data["sand"] = {"means": aux2, "sigmas": aux1}

def saveParameters(dataIn):
    with open("data.json", "w+") as fp:
        json.dump(dataIn, fp)
```

