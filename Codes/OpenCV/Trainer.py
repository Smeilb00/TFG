from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle


def run(emb, recon, inputle, kernel):
    # Cargamos los embeddings generados antes
    data = pickle.loads(open(emb, "rb").read())
    le = LabelEncoder()
    labels = le.fit_transform(data["nombres"])

    # Ahora entrenemos el modelo usando los embeddings para crear el sistema de reconocimiento
    recognizer = SVC(C=100, kernel=kernel, probability=True, gamma=1)
    recognizer.fit(data["embeddings"], labels)

    # Guardamos el sistema entrenado
    f = open(recon, "wb")
    f.write(pickle.dumps(recognizer))
    f.close()

    # Guardamos el label encoder también
    f = open(inputle, "wb")
    f.write(pickle.dumps(le))
    f.close()