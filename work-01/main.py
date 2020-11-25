import matplotlib.pyplot as plt
import numpy as np
import json

print("Se han cargado las libreras correctamente")

with open('data.json', 'r+') as fp:
    data = json.load(fp)

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
    suma = 0
    for i in range(m):
        for j in range(n):
            count += (img[i][j]-mm)**2
    print(np.sqrt(count/(n*m)))
    return list(np.sqrt(count/(n*m))), mm


def pdf(m, o, pixel):
    return (1/np.sqrt(2*np.pi*(o**2)))*((np.e)**(-((pixel-m)**2)/(2*(o**2))))

if False:
    aux1, aux2 = sigma(sea)
    data["sea"] = {"means": aux2, "sigmas": aux1}
    aux1, aux2 = sigma(sand)
    data["sand"] = {"means": aux2, "sigmas": aux1}

def saveParameters(dataIn):
    with open("data.json", "w+") as fp:
        json.dump(dataIn, fp)

print("Los parámetros de la imagen son: " + str(data["sea"]))
print("Los parámetros de la imagen son: " + str(data["sand"]))

plt.ion()
plt.imshow(sea_sand)

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

fig, ax = plt.subplots(2,2)

ax[0,0].imshow(sea)
ax[0,1].imshow(transpuesta(sand))
ax[1,1].imshow(filtro(sea_sand, data))
ax[1,0].imshow(sea_sand)
input("Finish?")