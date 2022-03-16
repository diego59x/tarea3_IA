import numpy as np
import matplotlib.pyplot as plt 
import pdb
import imageio
import shutil
import os 

# Generar matriz de distancia
def getDismat(points):
    dismat = np.zeros((N,N))
    for i in range(N):
        for j in range(i,N):
            dismat[i][j] = dismat[j][i] = np.linalg.norm(points[i]-points[j])
    return dismat

def init():
    alpha = 0.9
    t = (1,100)
    TIME = 1000
    way = np.arange(N)
    waydis = calWayDis(way)
    return alpha,t,TIME,way,waydis

# Calcular la longitud del camino
def calWayDis(way0):
    waydis = 0
    for i in range(N-1):
        waydis +=dismat[way0[i]][way0[i+1]] 
    waydis += dismat[way0[N-1]][way0[0]]
    return waydis

def draw(way,dist,nodo_inicial):
    global N,points ,TIMESIT, PNGFILE, PNGLIST
    plt.cla()
    plt.title('cross=%.4f' % dist)
    xs = [points[i][0] for i in range(N)]
    ys = [points[i][1] for i in range(N)]
    plt.scatter(xs, ys, color='b')
    xs = np.array(xs)
    ys = np.array(ys)
    # plt.plot(xs[[0, solutionpath[0]]], ys[[0, solutionpath[0]]], color='r')
    # Ruta de conexión
    for i in range(N-1):
        plt.plot(xs[[way[i], way[i + 1]]], ys[[way[i], way[i + 1]]], color='r')
    # Conecte el punto final con el punto inicial
    plt.plot(xs[[way[N - 1], 0]], ys[[way[N - 1], 0]], color='r')
    for i, p in enumerate(points):
        plt.text(*p, '%d' % i)
    plt.savefig('%s/%d.png' % (PNGFILE, TIMESIT))
    PNGLIST.append('%s/%d.png' % (PNGFILE, TIMESIT))
    TIMESIT += 1

nodo_inicial = int(input("Ingrese el numero del nodo inicial: ")) - 1

file = './Ciudades.txt'
c = []
with open(file, "r") as file:
    for line in file.readlines():
        line = [int(x.replace("\n", "")) for x in line.split(" ")]
        c.append(line)

    n = c[0][0]
    c.pop(0)
    

first_element = c[0]
c[0] = c[nodo_inicial]
c.pop(nodo_inicial)
c.append(first_element)

points = np.array(c)

print(points)

N = points.shape[0]
dismat = getDismat(points)
alpha,t,TIME,way,waydis=init()
t0 = t[1]
K=0.8

# Dibujo inicial
TIMESIT = 0
PNGFILE = './png/'
PNGLIST = []
if not os.path.exists(PNGFILE):
    os.mkdir(PNGFILE)
else:
    shutil.rmtree(PNGFILE)
    os.mkdir(PNGFILE)
# Registre el resultado de cada iteración
result = []
tempway = way.copy()
bestway = way.copy()
bestdis = 10000

while t0>t[0]:
    for i in range(TIME):
        if np.random.rand() > 0.5:
        # Intercambio de dos puntos
            while True:
            # Genera aleatoriamente 2 puntos diferentes,
                city1 = int(np.ceil(np.random.rand()*(N-1)))
                city2 = int(np.ceil(np.random.rand()*(N-1)))
                if city1!=city2:
                    break
            tempway[city1],tempway[city2]=tempway[city2],tempway[city1]
        else:
            # 3 puntos
            while True:
                city1 = int(np.ceil(np.random.rand()*(N-1)))
                city2 = int(np.ceil(np.random.rand()*(N-1))) 
                city3 = int(np.ceil(np.random.rand()*(N-1)))
                if((city1 != city2)&(city2 != city3)&(city1 != city3)):
                    break
            # Las siguientes tres sentencias hacen que city1 <city2 <city3
            if city1 > city2:
                city1,city2 = city2,city1
            if city2 > city3:
                city2,city3 = city3,city2
            if city1 > city2:
                city1,city2 = city2,city1
            #Las siguientes tres líneas de código insertan los datos en el intervalo (ciudad1, ciudad2) después de ciudad3
            temp = tempway[city1:city2].copy()
            tempway[city1:city3-city2+1+city1] = tempway[city2:city3+1].copy()
            tempway[city3-city2+1+city1:city3+1] = temp.copy()

        tempdis = calWayDis(tempway)
        if tempdis<waydis:
            way = tempway.copy()
            waydis = tempdis
        if tempdis<bestdis:
            bestway = tempway.copy()
            bestdis = tempdis
            draw(bestway,bestdis,nodo_inicial)
        else:
            if np.random.rand()<np.exp(-(tempdis-waydis)/(t0)):
                way = tempway.copy()
                waydis = tempdis
                # Actualizar ruta
            else: tempway = way.copy()

    t0 *= alpha
    result.append(bestdis)

print("Ruta más corta:% s"%np.array(result[-1]))

print('Pasos:')
for i in bestway:
    print(i,end=" --> ")
print(bestway[0])


#Generation path gif 
generated_images = []
for png_path in PNGLIST:
    generated_images.append(imageio.imread(png_path))
# shutil.rmtree (PNGFILE) # se puede eliminar
generated_images = generated_images + [generated_images[-1]] * 10
imageio.mimsave('Sovle_TSP01_0.gif', generated_images, 'GIF', duration=0.1)

