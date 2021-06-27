import os
import cv2
from imutils import paths
import imutils
import numpy
import pickle


def run(pathDetector, model, datapool, umbralConfianza, pathEmbeddings):
    # Cargamos en base a los archivos precargados el detector
    protoPath = os.path.sep.join([pathDetector, "deploy.prototxt"])
    modelPath = os.path.sep.join([pathDetector,
                                  "res10_300x300_ssd_iter_140000.caffemodel"])
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    # Ahora cargamos el sistema de reconocimiento
    embedder = cv2.dnn.readNetFromTorch(model)

    # Generamos una lista con las imagenes en el datapool
    imagePaths = list(paths.list_images(datapool))

    # Generamos 2 listas además para guardar los nombres y sus correspondientes embeddings
    # Junto con el total de nombres
    embeddings = []
    nombres = []
    total = 0
    for (i, imagePath) in enumerate(imagePaths):
        # Extraemos el nombre de cada persona asociada a la imagen
        nomb= imagePath.split(os.path.sep)[-2]
        nombre = nomb.split("/").pop()

        # Para cada imagen, la cargamos y la reescalamos en ancho para obtener las dimensiones finales
        imagen = cv2.imread(imagePath)
        imagen = imutils.resize(imagen, width=600)
        (h, w) = imagen.shape[:2]

        # Para cada imagen generamos su "blob"
        imageBlob = cv2.dnn.blobFromImage(cv2.resize(imagen, (300, 300)), 1.0, (300, 300),
                                          (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # Localizamos caras en las imagenes usando OpenCV
        detector.setInput(imageBlob)
        resultados = detector.forward()

        # Tiene que detectar minimo una cara en la imagen para seguir
        if len(resultados) > 0:
            # Suponemos que cada imagen del datapool tiene solo una cara
            # Escogemos la que tiene mayor probabilidad entonces
            i = numpy.argmax(resultados[0, 0, :, 2])
            confianza = resultados[0, 0, i, 2]

            # Esta detección tiene que tener un valor de confianza mayor que el umbral
            if confianza > umbralConfianza:
                # Cargamos las cordenas extraidas
                box = resultados[0, 0, i, 3:7] * numpy.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Extramos los datos ROI de la imagen con sus dimensiones
                cara = imagen[startY:endY, startX:endX]
                (fH, fW) = cara.shape[:2]

                # Tiene que tener unas dimensiones minimas
                if fW < 20 or fH < 20:
                    continue
                # Construimos el "blob" de las detecciones ROI y lo pasamos por le modelo
                # para obtener la cuantificacion de la cara en 128
                blob = cv2.dnn.blobFromImage(cara, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(blob)
                vec = embedder.forward()

                # Añadimos ahora el nombre junto con su embedding las listas y sumamos 1 al total
                nombres.append(nombre)
                embeddings.append(vec.flatten())
                total += 1

            # Ahora para guardarlo en un fichero construimos un diccionario en el que guardamos ambos arrays

    data = {
        "embeddings": embeddings,
        "nombres": nombres
        }
    f = open(pathEmbeddings, "wb")
    f.write(pickle.dumps(data))
    f.close()
