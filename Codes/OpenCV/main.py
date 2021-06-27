from tkinter import *
from tkinter import filedialog
from tkinter import ttk
import AnalizeDatapool, Trainer, Recognize
import time, datetime, os
from PIL import Image
from PIL import ImageTk

rutaInput = ""
rutaDatapool = "../Datapool"
rutaOutput = "../Test Final - Interfaz Grafica/Output"


def ventanaTiempos(ts):
    root = Tk()
    root.title("Face Recognition")
    root.geometry("390x260")
    root.resizable(0, 0)
    root.iconbitmap("..\Config\proconsi.ico")

    tiempos = ttk.Treeview(root, columns="ts")
    tiempos.heading("#0", text="Tiempos")
    tiempos.heading("ts", text="Tiempo en segundos")
    tiempos.column("#0", minwidth=0, width=220)
    tiempos.column("ts", minwidth=0, width=130)
    item = tiempos.insert("", "end", text="Tiempo inicial: ", values=str(ts["t0"]))
    item = tiempos.insert("", "end", text="Leer datapool: ", values=str(ts["t1"]))
    item = tiempos.insert("", "end", text="Entrenar el sistema: ", values=str(ts["t2"]))
    item = tiempos.insert("", "end", text="Reconocer caras dentro de la imagen: ", values=str(ts["t3"]))
    tiempos.place(x=20, y=20)

    root.mainloop()


def uploadInput():
    global rutaInput
    path = filedialog.askopenfilename(initialdir=os.getcwd()+"/../")
    rutaInput = path


def uploadDatapool():
    path = filedialog.askdirectory(initialdir=os.getcwd()+"/../")
    rutaDatapool = path


def uploadOutput():
    path = filedialog.askdirectory(initialdir=os.getcwd()+"/../")
    rutaOutput = path
    print(rutaOutput)


def start(kernel):
    if(rutaInput == ""):
        alertBoxErrors(1)
    elif(kernel.get() == 0):
        alertBoxErrors(3)
    else:
        rutain = rutaInput.split('.')
        ext = rutain[len(rutain)-1]
        validas = ["bmp", "tiff", "webp", "jpg", "png", "jpeg"]
        if(ext not in validas):
            alertBoxErrors(2)
        else:
            t0 = time.time()
            tiempos = {"t0": datetime.datetime.now().strftime("%H:%M:%S"), "t1": "", "t2": "", "t3": ""}
            if not os.path.exists(rutaOutput + "/embeddings.pickle"):
                AnalizeDatapool.run("Face_Detection_Model/", "openface.nn4.small2.v1.t7", rutaDatapool + "/", 0.95,
                                    rutaOutput + "/embeddings.pickle")
            tiempos["t1"] = time.time() - t0
            t0 = time.time()
            if kernel.get() == 1:
                kern = "linear"
            elif kernel.get() == 2:
                kern = "rbf"
            else:
                kern = "poly"
            Trainer.run(rutaOutput + "/embeddings.pickle", rutaOutput + "/recognizer.pickle", rutaOutput + "/le.pickle", kern)
            tiempos["t2"] = time.time() - t0

            t0 = time.time()
            tfinal = Recognize.run("face_detection_model/", "openface.nn4.small2.v1.t7", rutaOutput + "/recognizer.pickle", rutaOutput + "/le.pickle",
                                   rutaInput, 0.95, rutaOutput, t0)
            tiempos["t3"] = tfinal
            if(tfinal == -1):
                alertBoxErrors(4)
            else:
                ventanaTiempos(tiempos)


def alertBoxAyuda():
    root = Tk()
    root.title("Ayuda")
    root.geometry("370x310")
    root.resizable(0, 0)
    root.iconbitmap("..\Config\proconsi.ico")

    lblAyuda = Label(root, text="Bienvenido a la ayuda del programa")
    lblAyuda.pack(side=TOP, pady=10)

    lbltxt = Label(root, text="El primer paso es seleccionar la imagen  sobre la que quiere realizar el reconocimiento. "
                              "Los formatos aceptados son: JPG, PNG, JPEG, TIFF, WEBP y BMP.", wraplength=350, justify=LEFT)
    lbltxt.pack(side=TOP, anchor="nw", padx=10)

    lbltxt2 = Label(root,
                   text="La aplicación cuenta con la opción de modificar la ruta en la que se encuentra el set de datos "
                        "que se usará para entrenar el sistema, si no desea modificarlo se utilizará el propio."
                    , wraplength=350, justify=LEFT)
    lbltxt2.pack(side=TOP, anchor="nw", padx=10)

    lbltxt3 = Label(root,
                    text="De la misma manera, también cuenta con la opción de especificar la ruta en la que se guardarán "
                         "tanto los archivos de configuración como la imagen de salida (con los rostros detectados), si"
                         "no se modifica se utilizará la propia ruta del sistema."
                    , wraplength=350, justify=LEFT)
    lbltxt3.pack(side=TOP, anchor="nw", padx=10)

    lbltxt4 = Label(root,
                    text="Por último, podrá seleccionar el algoritmo que se desea utilizar para el reconocimiento, NO "
                         "existe opción por defecto, deberá seleccionar uno."
                    , wraplength=350, justify=LEFT)
    lbltxt4.pack(side=TOP, anchor="nw", padx=10)

    btnOK = Button(root, text="Salir", command=lambda: close(root))
    btnOK.place(x=170, y=270)

    root.mainloop()


def alertBoxErrors(tipoError):
    root = Tk()
    root.title("ERROR")

    root.resizable(0, 0)
    root.iconbitmap("..\Config\proconsi.ico")
    if(tipoError == 1):
        root.geometry("410x100")
        lblAyuda = Label(root, text="Debes seleccionar una imagen sobre la que realizar el reconocimiento.")
        lblAyuda.place(x=20, y=20)
    elif(tipoError == 2):
        root.geometry("380x100")
        lblAyuda = Label(root, text="El formato del objeto introducido no corresponde a una imagen.")
        lblAyuda.place(x=20, y=20)
    elif(tipoError == 3):
        root.geometry("430x100")
        lblAyuda = Label(root, text="No se ha seleccionado el algoritmo que se utilizará para el reconocimiento.")
        lblAyuda.place(x=20, y=20)
    elif (tipoError == 4):
        root.geometry("325x100")
        lblAyuda = Label(root, text="No existen caras conocidas en la imagen introducida.")
        lblAyuda.place(x=20, y=20)

    btnOK = Button(root, text="Aceptar", command=lambda :close(root))
    btnOK.place(x=180, y=65)

    root.mainloop()


def close(root):
    root.destroy()


def main():
    root = Tk()
    root.title("Face Recognition")
    root.geometry("370x200")
    root.resizable(0, 0)
    root.iconbitmap("..\Config\proconsi.ico")

    # Añadimos objetos que utilizaremos después
    labelInput = Label(root, text="Imagen en la que se quiere realizar el reconocimiento:")
    labelInput.place(x=20, y=20)

    butUploadFile = Button(root, text="Select", command=uploadInput)
    butUploadFile.place(x=310, y=20)

    labelOutput = Label(root, text="Directorio en el que se guardará el resultado:")
    labelOutput.place(x=20, y=50)
    butUploadFolder1 = Button(root, text="Select", command=uploadOutput)
    butUploadFolder1.place(x=310, y=50)

    labelDatapool = Label(root, text="Directorio en el que se encuentra el datapool:")
    labelDatapool.place(x=20, y=80)
    butUploadFolder2 = Button(root, text="Select", command=uploadDatapool)
    butUploadFolder2.place(x=310, y=80)

    labelDatapool = Label(root, text="Algoritmo que se desea utilizar:")
    labelDatapool.place(x=20, y=110)

    kernel = IntVar()

    radioLinear = Radiobutton(root, text="Linear", variable=kernel, value=1)
    radioLinear.place(x=200, y=110)

    radioRBF = Radiobutton(root, text="RBF", variable=kernel, value=2)
    radioRBF.place(x=260, y=110)

    radioRBF = Radiobutton(root, text="Poly", variable=kernel, value=3)
    radioRBF.place(x=305, y=110)

    imgHelp = Image.open("../Config/ayuda.png")
    imgHelp = imgHelp.resize((25, 25), Image.ANTIALIAS)
    imageHelp = ImageTk.PhotoImage(imgHelp)

    buttHelp = Button(root, image=imageHelp, compound=LEFT, command=lambda :alertBoxAyuda()).place(x=321, y=155)


    buttStart = Button(root, text="Iniciar Reconocimiento", command=lambda :start(kernel)).place(x=120, y=160)

    root.mainloop()

main()
