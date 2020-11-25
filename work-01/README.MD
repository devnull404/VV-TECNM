# Nombre
Vladimir Villegas Luis.  

# Carrera
Ingeniería Informática, 7º Semestre.  

# Materia
Machine Learning.

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

# Función para procesar cada pixel de la imagen TEST
Finalmente, debemos utilizar los parámetros obtenidos de cada imagen de entrenamiento para aplicarlos a la imagen TEST, y guardar los resultados en otra imagen auxiliar, aplicando las condicionales necesarias para saber si se trata de **mar** o **arena**

```python
def filtro(img, data):
    n, m, c = img.shape
    aux = np.empty((n, m, c), dtype=int)
    aux[:][:][:] = 0
    for i in range(n):
        for j in range(m):
            pixel = img[i][j][2]
            if pdf(data["sea"]["means"][0], data["sea"]["sigmas"][0], pixel) > 0.005:
                aux[i][j][:] = 254
    return aux
```

# Mostrando los resultados

Después de haber realizardo todos los cálculos, podemos mostrar nuestros resultados en una sola figura con subplots, mostrando todas las imágenes (sea, sand, sea_sand y test).

```python
fig, ax = plt.subplots(2,2)
ax[0,0].imshow(sea)
ax[0,1].imshow(transpuesta(sand))
ax[1,1].imshow(filtro(sea_sand, data))
ax[1,0].imshow(sea_sand)
```

