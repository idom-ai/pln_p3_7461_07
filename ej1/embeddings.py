"""
embeddings.py - EJERCICIO 1: de reviews a secuencias de índices, y matriz de embeddings preentrenados.
"""
from collections import Counter

import numpy as np


def tokenizar(reader, fileids):
    """
    Devuelve los tokens de cada review, en minúsculas.
    No se lematiza ni se quitan stopwords: GloVe tiene las palabras tal cual y 'not' importa.
    """
    textos = []
    for review in reader.paras(fileids=list(fileids)):
        tokens = [palabra.lower() for parrafo in review for frase in parrafo for palabra in frase]
        textos.append(tokens)
    return textos


class Vectorizador(object):
    """
    Convierte cada review (lista de tokens) en una lista de LONGITUD índices.
    Índice 0 = relleno, índice 1 = palabra desconocida, 2, 3... = palabras del vocabulario.
    """

    def __init__(self, longitud=100, min_frecuencia=2):
        self.longitud = longitud                # tokens por review
        self.min_frecuencia = min_frecuencia    # las palabras más raras pasan a "desconocida"

    def fit(self, textos):
        """Aprende el vocabulario con las reviews de entrenamiento."""
        frecuencias = Counter(palabra for tokens in textos for palabra in tokens)
        self.vocabulario = {}
        for palabra, n in frecuencias.most_common():
            if n >= self.min_frecuencia:
                self.vocabulario[palabra] = len(self.vocabulario) + 2
        return self

    def transform(self, textos):
        """Devuelve una matriz (reviews x longitud) de índices; lo que sobra se rellena con 0."""
        X = np.zeros((len(textos), self.longitud), dtype='int64')
        for i, tokens in enumerate(textos):
            indices = [self.vocabulario.get(palabra, 1) for palabra in tokens[:self.longitud]]
            X[i, :len(indices)] = indices
        return X


def matriz_embeddings(vectorizador, glove):
    """
    Matriz (palabras x dimensión) para la capa Embedding: la fila i es el vector GloVe de la
    palabra con índice i. Las palabras que no están en GloVe se quedan con un vector aleatorio,
    y el relleno (fila 0), con ceros.
    """
    n_palabras = len(vectorizador.vocabulario) + 2
    matriz = np.random.RandomState(42).normal(0, 0.1, (n_palabras, glove.vector_size))
    matriz[0] = 0
    for palabra, i in vectorizador.vocabulario.items():
        if palabra in glove:
            matriz[i] = glove[palabra]
    return matriz.astype('float32')
