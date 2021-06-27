import cv2
import face_recognition
import os
import pickle
import time

from imutils import paths


def run(data, tipoEjec):
    imagen = data["Input"]
    input = data["Input"] + "/../"
    output = data["Output"]
    datapool = data["Datapool"]
    config = data["Config"]

    # Path de la carpeta padre
    tBase = time.time()
    # Primero comprobamos si ya se han generado los encodings previamente del datapool.
    if not os.path.exists(config + "\\saved_encodings.dat"):
        faceEncodings = gen_Encodings(datapool, output)
        t1 = time.time() - tBase
    else:
        with open(config + "\\saved_encodings.dat", "rb") as f:
            faceEncodings = pickle.load(f)
            t1 = time.time() - tBase

    # Definimos una lista de nombres conocidos en base al datapool.
    # Esto limita a que en el datapool las carpetas se tengan que llamar con el nombre que queramos mostrar.
    names = gen_NameList(datapool)

    # Guardamos una fuente para escribir en el cuadrado
    font = cv2.FONT_HERSHEY_COMPLEX

    # Cargamos la imagen en la que queremos reconocer caras
    # Primero la pasamos a blanco y negro
    bw = cv2.cvtColor(cv2.imread(imagen), cv2.COLOR_BGR2GRAY)
    # Si quisieramos guardarla antes de realizar el reconocimiento:
    cv2.imwrite(input + "\\" + "inputBW.jpg", bw)

    # La cargamos al face_recognition
    img = face_recognition.load_image_file(input + "\\" + "inputBW.jpg")

    # Creamos un diccionario donde almacenaremos los arrays que necesitaremos
    recon = {}
    loc = face_recognition.face_locations(img)
    enc = face_recognition.face_encodings(img, loc)
    nom = find_Faces(datapool, enc, faceEncodings)
    nom2 = nom.copy()
    nom2.pop()
    recon.setdefault("loc", loc)
    recon.setdefault("enc", enc)
    recon.setdefault("nom", nom2)

    # Dibujamos los cuadrados en la imagen
    draw(img, font, recon.get("loc"), recon.get("nom"))
    if tipoEjec == 0:
        cv2.imshow("Output", img)
    cv2.imwrite(output + "\\Output.jpg", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return nom

def gen_Encodings(datapool, output):
    # Este método genera encodings para cada en el datapool y exporta el resultado final
    encodings = []
    i = 0
    images = []
    for person in os.listdir(datapool):
        direc = list(paths.list_images(datapool + "\\" + person))
        for image in direc:
            images.append(face_recognition.load_image_file(image))
            # Aquí se generan los encodings
            encodings.append(face_recognition.face_encodings(images[i])[0])
            i += 1

    # Lo exportamos
    with open(output + "\\saved_encodings.dat", "wb") as f:
        pickle.dump(encodings, f)
    return encodings


def gen_NameList(datapool):
    # Este método genera una lista de nombres conocidos en función de
    # las carpetas existentes en la carpeta raíz datapool
    t2 = time.time()
    namelist = []
    for carpeta in datapool:
        namelist.append(carpeta)

    return namelist


def find_Faces(datapool, enc_input, enc_datapool):
    nom = []
    name = ""
    totalImages = list(paths.list_images(datapool)).__len__()
    t3 = time.time()
    for enc in enc_input:
        # Generamos un Array de Trues y Falses donde cada True representa si se ha encontrado.
        coinc = face_recognition.compare_faces(enc_datapool, enc, tolerance=0.44)
        # Ahora comprobamos si se ha encontrado y si es asi guardamos su nombre
        if True in coinc:
            found = coinc.index(True)
            index = 0
            for carpeta in os.listdir(datapool):
                 if not os.path.isfile(carpeta):
                    len = os.listdir(datapool + "\\" + carpeta).__len__()
                    found = found - len
                    if found < 0:
                        name = carpeta.__str__()
                        break
        else:
            name = "Unknown"
        nom.append(name)
    nom.append(time.time()-t3)
    return nom


def draw(img, font, loc, nom):
    for (top, right, bottom, left), name in zip(loc, nom):

        # Cambiar el color segun el nombre:
        if name != "???":
            color = (0, 255, 0)  # Verde
        else:
            color = (0, 0, 255)  # Rojo

        # Dibujar los recuadros alrededor del rostro:
        cv2.rectangle(img, (left, top), (right, bottom), color, 2)
        cv2.rectangle(img, (left, bottom - 20), (right, bottom), color, -1)

        # Escribir el nombre de la persona:
        cv2.putText(img, name, (left, bottom - 6), font, 0.6, (0, 0, 0), 1)