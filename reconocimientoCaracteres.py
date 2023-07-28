import cv2 
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm



def get_hog():
    winSize = (20,20)
    blockSize = (8,8)
    blockStride = (4,4)
    cellSize = (8,8)
    nbins = 9 
    derivAperture = 1 # convergence's parameters
    winSigma = 2. # noise filter
    histogramType = 0
    L2HysThreshold= 0.2 # constant of normalization 
    gammaCorrection = 1 
    nlavels = 64 #defeaut 
    signedGradient = True # include sign in gradient 
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,histogramType, L2HysThreshold, gammaCorrection, nlavels,signedGradient)
    return hog 

def escalar(img, m, n):
    if m > n:
        imgN = np.uint8(255 * np.ones((m, int(round((m - n) / 2)), 3)))
        escalada = np.concatenate((np.concatenate((imgN, img), axis=1), imgN), axis=1)
    else:
        imgN = np.uint8(255 * np.ones((int(round((n - m) / 2)), n, 3)))
        escalada = np.concatenate((np.concatenate((imgN, img), axis=0), imgN), axis=0)

    img = cv2.resize(escalada, (20, 20))
    return img

def obtenerDatos():
    posibleEtiquetas = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                        'A', 'B', 'C', 'D', 'E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z']

    datos = []
    etiquetas = []

    for i in range(1,26): 
        for j in posibleEtiquetas: 
            img = cv2.imread(j+'-'+str(i)+".jpg")
            if img is not None: 
                m,n, _ =  img.shape
                if m !=20 or n!=20: 
                    img = escalar(img, m,n)
                etiquetas.append(np.where(np.array(posibleEtiquetas)==j)[0][0])  
                hog = get_hog() # inicializa
                datos.append(np.array(hog.compute(img)))

    datos = np.array(datos)
    etiquetas = np.array(etiquetas)
    return datos, etiquetas

# Implementemos un clasificador no - lineal
def clasificadorCaracteres(): 
    datos, etiquetas = obtenerDatos()
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(datos, etiquetas) # entrena
    SVM = svm.SVC(kernel='linear', probability=True, random_state=0, gamma='auto')
    SVM.fit(datos, etiquetas)
    return knn, SVM







#datos, etiquetas = obtenerDatos()
#
## Separamos los datos de entrenamiento de los de prueba.
#
#X_train, X_test, Y_train, Y_test = train_test_split(datos, etiquetas, test_size=0.2, random_state=np.random)
#
#knn = KNeighborsClassifier(n_neighbors=1)
#knn.fit(X_train, Y_train) # entrena
#errorEntrenamientoKnn = (1 - knn.score(X_train, Y_train))*100 # score -- efectividad
#print("El error de entrenamiento es del knn es:"+str(round(errorEntrenamientoKnn, 2)) + "%")
#errorPruebaKnn = (1 - knn.score(X_test, Y_test))*100
#print("El error de Prueba del knn es:"+str(round(errorPruebaKnn, 2)) + "%")



        