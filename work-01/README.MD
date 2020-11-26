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
El módulo **JSON** es utilizado para guardar variables en el disco duro, y esto es útil debido a que este código trabaja con imágenes para calcular los parámetros estadísticos, entonces, una vez que se hayan calculado, no es necesario calcularlos de nuevo, ya que se guardarán en el archivo **.json** en forma de diccionario, luego, en la función PDF sólo utilizaremos los parámetros obtenidos, sin tener que volver a llamar a las funciones para que recalculen los datos que necesitamos. 

```python
with open('data.json', 'r+') as fp:
    #Cargamos el archivo data.json a una variable
    data = json.load(fp)
```

# Leyendo las imágenes
Después de haber cargado todas las librerías necesarias, podemos empezar cargando las imágenes para empezar a trabajar con ellas utilizando la librería **MatplotLib**, que llamamos en la primer sección como **plt**, entonces, cargamos en tres variables cada imagen con la función **plt.imread()**:

```python
# Cargamos la imagen a sus respectivas variables
sea = plt.imread("./img/sea.jpg")
sand = plt.imread("./img/sand.jpg")
sea_sand = plt.imread("./img/sea_sand.jpg")
```

# Creando función para transponer una imagen dada
Como sería ideal que todas las imágenes tuvieran el mismo tamaño y orientación, en este caso una se entuentra rotada, por lo que es necesario realizar una transformación, que en este caso, funcionaría una transpuesta para pasar de **(m,n)** => **(n,m)**, que podría realizarse con la librería **NUMPY**, pero haremos de forma manual para entender bien el procedimiento. El código que se muestra a continuación crea una imagen auxiliar vacía con dimensiones invertidas **np.empty(b, a, c)** obteniendo los valores con el método **.shape**, que nos da las dimensiones de la imagen de entrada en la función. Finalmente, se pasa por cada pixel con dos ciclos *For* anidados para pasar por cada fila y collumna, para intercambiar las comumnas por las filas mediante **aux[j][i] = img[i][j]**

```python
def transpuesta(img):
    # Buscamos las dimensiones de la imagen
    a, b, c = img.shape
    aux = np.empty((b, a, c), dtype=int)
    # Creamos una matriz vacía con tamaño transpuesto
    for i in range(a):
        for j in range(b):
            # Aplicamos la definición de transpuesta
            aux[j][i] = img[i][j]
    # Devolvemos el resultado
    return aux
```
# Función para calcular el mean (µ) de cada imagen
Podemos realizarlo de manera manual, calculando por fórmula, pasando por cada pixel en cada imagen y encontrar sus means, por lo que de la misma manera que en el segmento de código anterior, utilizamos un ciclo **For** anidado, pasando por cada pixel y guardando la suma de cada uno en un contador, que al final se agrega a la fórmula para regresar el resultado.

```python
def mu(img):
    # Obtenemos las dimensiones de la imagen de entrada
    a, b, c = img.shape
    # Creamos un contador para cada canal de la imagen (R,G,B)
    count = [0, 0, 0]
    # Pasamos por cada pixel
    for i in range(a):
        for j in range(b):
            # Sumamos
            count[0] += img[i][j][0]
            count[1] += img[i][j][1]
            count[2] += img[i][j][2]
    # Dividimos entre el número total de pixeles
    count[0], count[1], count[2] = [count[0]/(a*b), count[1]/(a*b), count[2]/(a*b)]
    # Devolvemos el resultado
    return count
```

# Función para calcular Sigma (σ) 
Por otra parte, generamos otra función que calcula el σ de cada imagen, y tiene como datos de entrada los valores µ calculados de cada imagen, por lo que sólo es necesario llamar a esta para obtener los dos valores (µ y σ).

```python
def sigma(img):
    # Llamamos a nuestra función mu para
    # Calcular los means y entregarlos 
    # aquí donde sean necesarios
    mm = mu(img)
    # Obtenemos las dimensiones de la imagen
    m, n, c = img.shape
    # Creamos un contador
    count = 0
    suma = 0
    # Pasamos por cada pixel de la imagen de entrada
    for i in range(m):
        for j in range(n):
            # Utilizamos la definición de σ
            count += (img[i][j]-mm)**2
    print(np.sqrt(count/(n*m)))
    # Regresamos el valor
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
# Llamamos a la función sigma para obtener los parámetros
# mu y sigma, y se los asignamos a una parte del diccionario (SEA).
aux1, aux2 = sigma(sea)
data["sea"] = {"means": aux2, "sigmas": aux1}
# Llamamos a la función sigma nuevamente para obtener los parámetros
# mu y sigma, y se los asignamos a otra parte del diccionario (SAND).
aux1, aux2 = sigma(sand)
data["sand"] = {"means": aux2, "sigmas": aux1}

# Con esta función guardamos los valores de data en el disco duro
# en el archivo data.json para que puedan ser llamados incluso
# después de que se haya ejecutado el código.
def saveParameters(dataIn):
    with open("data.json", "w+") as fp:
        json.dump(dataIn, fp)
```

# Función para procesar cada pixel de la imagen TEST
Finalmente, debemos utilizar los parámetros obtenidos de cada imagen de entrenamiento para aplicarlos a la imagen TEST, y guardar los resultados en otra imagen auxiliar, aplicando las condicionales necesarias para saber si se trata de **mar** o **arena**

```python
def filtro(img, data):
    # Obtenemos como entrada la imagen, y los parámetros
    # estadísticos de SAND y SEA.

    # Obtenemos las dimensiones de la imagen de entrada
    n, m, c = img.shape
    # Creamos una imagen auxiliar (matriz de (n,m,3)) para
    # guardar los resultados.
    aux = np.empty((n, m, c), dtype=int)
    # Los mandamos a 0.
    aux[:][:][:] = 0
    # Pasamos por cada pixel de la imagen SEA_SAND.
    for i in range(n):
        for j in range(m):
            # Obtenemos el valor del pixel.
            pixel = img[i][j][2]
            # Aplicamos las condiciones que nos explica en el PDF
            if pdf(data["sea"]["means"][0], data["sea"]["sigmas"][0], pixel) > 0.005:
                # Si es mayor que un humbral, lo mandamos a 254
                # que significa blanco.
                aux[i][j][:] = 254
    # Regresamos la matriz auxiliar donde se encuentran los
    # resultados.
    return aux
```

# Mostrando los resultados

Después de haber realizardo todos los cálculos, podemos mostrar nuestros resultados en una sola figura con subplots, mostrando todas las imágenes (sea, sand, sea_sand y test).

```python
# Utilizamos la librería matplotlib.pyplot para graficas
# nuestros resultados.

#plt.ion() Nos permite trabajar de manera interactiva.
plt.ion()
# Creamos un subplot de (2,2)
fig, ax = plt.subplots(2,2)
# Graficamos en el respectivo lugar.
ax[0,0].imshow(sea)
ax[0,1].imshow(transpuesta(sand))
ax[1,1].imshow(filtro(sea_sand, data))
ax[1,0].imshow(sea_sand)
```

# Resultados del algoritmo:

![Resultado Binario]("./img/RESULT.png")
![Resultado Binario]("./img/mozaico.png")
