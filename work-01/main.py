import matplotlib.pyplot as plt
import numpy as np

print("Se han cargado las libreras correctamente")

sea, sand, sea_sand = [plt.imread("./img/sea.jpg"), plt.imread("./img/sand.jpg"), plt.imread("./img/sea_sand.jpg")]

def transpuesta(img):
    a, b, c = img.shape
    aux = np.empty((b, a, c), dtype=int)
    for i in range(a):
        for j in range(b):
            aux[j][i] = img[i][j]
    return aux


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

def sigma(img):
    mm = mu(img)
    m, n, c = img.shape
    count = 0
    for i in range(m):
        for j in range(n):
            count += (img[i][j]-mm)**2
    return np.sqrt(count/(n*m)), mm

sigmafnc = [() for x in range(254)]


print("Valores estadísticos del set de entrenamiento: Sea -" + "\n" + str(sigma(sea)) + "\n\n")
print("Valores estadísticos del set de entrenamiento: Sand -" + "\n" + str(sigma(sand)) + "\n\n")
