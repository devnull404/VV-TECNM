import matplotlib.pyplot as plt
import numpy as np

print("Se han cargado las librerías correctamente")

I1, I2, I3 = [plt.imread("./img/3.jpg"), plt.imread("./img/2.jpg"), plt.imread("./img/3.jpg")]

I1_r = np.reshape(I1, (3,-1))