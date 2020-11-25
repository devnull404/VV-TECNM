# Código
Este código se basa en una ecuación gaussiana de densidad de probabilidad (PDF) con dos parámetros basados en el "entrenamiento" con dós imágenes para cada evento (1).

<img src="https://render.githubusercontent.com/render/math?math=P(A_{ij_{objetivo}}| \sigma_{entrenamiento},\mu_{entrenamiento})=\frac{1}{\sqrt{2 \pi \sigma ^2}}e^{\frac{(x-\mu)^2}{2\sigma^2}}">

Esto es la probabilidad de que el pixel en la imagen objetivo A(i,j) sea arena o mar.

El modelo de clasificación propuesto con los parámetros sigma y mu de cada set de entrenamiento, se puede encontrar en la siguiente imagen.

![alt text](https://github.com/devnull404/VV-TECNM/blob/master/work-01/img/modelo.jpg)

![Markdown Mark](/img/modelo.jpg)

En primera instancia, debemos instalar los módulos necesarios para poder trabajar con imágnes, una de estas es NUMPY, y MatplotLib, que nos permite graficar, entre muchas otras cosas, utilizando el código siguiente:  

```python
import matplotlib.pyplot as plt
import numpy as np
import json
```

Un módulo utilizado para aligerar la carga en el CPU para realizar pruebas muchas veces, es el JSON, que nos permite trabajar con formatos .json para guardar los datos parámetros que calculemos en el disco duro como un archivo denominado data.json para no realizar los mismos cálculos una y otra vez. Entonces, con las líneas de código siguientes podemos cargar al programa los datos contenidos en este archivo para llamarlos incluso después de que se haya ejecutado el programa:

```python
with open('data.json', 'r+') as fp:
    data = json.load(fp)
```
Después de haber cargado todas las librerías necesarias, podemos empezar cargando las imágenes para empezar a trabajar con ellas utilizando la librería **MatplotLib**, que llamamos en la primer sección como **plt**, entonces, cargamos en tres variables cada imagen con la función **plt.imread()**:

```python
sea, sand, sea_sand = [plt.imread("./img/sea.jpg"), plt.imread("./img/sand.jpg"), plt.imread("./img/sea_sand.jpg")]
```
