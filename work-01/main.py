import matplotlib.pyplot as plt
import numpy as np

print("Se han cargado las librerías correctamente")

I1, I2, I3 = [plt.imread("./img/3.jpg"), plt.imread("./img/2.jpg"), plt.imread("./img/3.jpg")]

n, m, c = I1.shape

I1_mu = np.mean(I1[:][:][0])

for i in range(n):
    for j in range(m):
        print(I1[i][j][0])

print(I1_mu)
