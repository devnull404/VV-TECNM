# Código
Este código se basa en una ecuación gaussiana de densidad de probabilidad (PDF) con dos parámetros basados en el "entrenamiento" con dós imágenes para cada evento (1).

<img src="https://render.githubusercontent.com/render/math?math=P(A_{ij_{objetivo}}| \sigma_{entrenamiento},\mu_{entrenamiento})=\frac{1}{\sqrt{2 \pi \sigma ^2}}e^{\frac{(x-\mu)^2}{2\sigma^2}}">

Esto es la probabilidad de que el pixel en la imagen objetivo A(i,j) sea arena o mar.

El modelo de clasificación propuesto con los parámetros sigma y mu de cada set de entrenamiento, se puede encontrar en la siguiente imagen.

![alt text](https://github.com/devnull404/VV-TECNM/blob/master/work-01/img/modelo.jpg)

![Markdown Mark](./img/modelo.jpg)

```python
import numpy as np  
for i in len(200):
    i
```