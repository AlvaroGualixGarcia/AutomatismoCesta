import spacy
import csv
import random
from spacy.training.example import Example

def cargar_datos(ruta_archivo):
    datos = []
    with open(ruta_archivo, newline='', encoding='utf-8') as csvfile:
        lector = csv.DictReader(csvfile)
        for fila in lector:
            texto = fila['name']
            anotaciones = []
            for producto in fila['name'].split(', '):
                inicio = texto.find(producto)
                fin = inicio + len(producto)
                anotaciones.append((inicio, fin, 'PRODUCTO'))
            datos.append((texto, {'entities': anotaciones}))
    return datos



def entrenar_modelo(nlp, datos_entrenamiento, iteraciones=1000000):
    ejemplos = []
    for texto, anotaciones in datos_entrenamiento:
        ejemplos.append(Example.from_dict(nlp.make_doc(texto), anotaciones))

    nlp.begin_training()
    for i in range(iteraciones):
        random.shuffle(ejemplos)
        for lote in spacy.util.minibatch(ejemplos, size=8):
            nlp.update(lote)
    return nlp

def procesar_pedido(nlp, texto):
    doc = nlp(texto)
    productos = []
    for ent in doc.ents:
        if ent.label_ == 'PRODUCTO':
            productos.append(ent.text)
    return productos

def main():
    # Paso 1: Cargar datos de entrenamiento desde el archivo CSV
    datos_entrenamiento = cargar_datos('prueba.csv')

    # Paso 2: Cargar el modelo de spaCy preentrenado para el idioma español
    nlp = spacy.blank('es')

    # Paso 3: Entrenar el modelo NER
    nlp = entrenar_modelo(nlp, datos_entrenamiento)

    # Paso 4: Procesar órdenes de los clientes
    texto = "Hola, quisiera dos Ginebra 15 botanicals Blumara y seis Helado After Dinner Magnum sin gluten 10 ud."
    productos_solicitados = procesar_pedido(nlp, texto)

    # Paso 5: Imprimir los productos solicitados
    print("Productos solicitados:")
    for producto in productos_solicitados:
        print(producto)

if __name__ == "__main__":
    main()
