"""
classifiers.py - EJERCICIOS 1 y 2: clasificadores de reviews con redes neuronales de PyTorch.

  - EmbeddingClassifier (ejercicio 1): MLP o RNN (ej1/arquitecturas/) con embeddings estáticos.
  - BertClassifier (ejercicio 2): ajuste fino (fine-tuning) de un BERT preentrenado pequeño.

Las dos redes se envuelven con skorch, que las convierte en clasificadores de scikit-learn
(fit / predict) y se encarga del bucle de entrenamiento.
"""
import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_class_weight
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from skorch.dataset import Dataset
from skorch.helper import predefined_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# GPU si la hay: CUDA (Windows con NVIDIA), MPS (Mac con Apple Silicon) o, si no, CPU
if torch.cuda.is_available():
    DISPOSITIVO = 'cuda'
elif torch.backends.mps.is_available():
    DISPOSITIVO = 'mps'
else:
    DISPOSITIVO = 'cpu'


class EmbeddingClassifier(ClassifierMixin, BaseEstimator):

    def __init__(self, red, matriz, parametros):
        self.red = red                # clase de la red: MLP o RNN
        self.matriz = matriz          # matriz de embeddings (embeddings.matriz_embeddings)
        self.parametros = parametros  # hiperparámetros de la red, p. ej. {'entrenable': True}

    def fit(self, X, y, X_val, y_val):
        """
        Entrena la red. X y X_val son matrices de índices (Vectorizador.transform).
        Se para cuando la pérdida en validación deja de bajar (early stopping).
        """
        # PyTorch necesita las clases como números: 'negativas' -> 0, 'neutras' -> 1, 'positivas' -> 2
        self.classes_ = np.unique(y)
        y = np.searchsorted(self.classes_, y)
        y_val = np.searchsorted(self.classes_, y_val)

        # Pesos de las clases ('balanced', como el mejor modelo de la P2)
        pesos = compute_class_weight('balanced', classes=np.unique(y), y=y)

        self.modelo_ = NeuralNetClassifier(
            self.red,
            module__matriz_embeddings=self.matriz,   # module__x -> argumento x de la red
            module__n_clases=len(self.classes_),
            **{'module__' + k: v for k, v in self.parametros.items()},
            criterion=torch.nn.CrossEntropyLoss,
            criterion__weight=torch.tensor(pesos, dtype=torch.float32),
            optimizer=torch.optim.Adam,
            lr=0.001,
            max_epochs=20,
            batch_size=64,
            iterator_train__shuffle=True,
            train_split=predefined_split(Dataset(X_val, y_val)),
            callbacks=[EarlyStopping(patience=3, load_best=True)],
            device=DISPOSITIVO,
        )
        self.modelo_.fit(X, y)
        return self

    def predict(self, X):
        """Devuelve la clase predicha ('negativas', 'neutras' o 'positivas') de cada review."""
        return self.classes_[self.modelo_.predict(X)]


class RedBert(torch.nn.Module):
    """BERT preentrenado con una capa de clasificación encima (la añade Hugging Face)."""

    def __init__(self, nombre, n_clases):
        super().__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(nombre, num_labels=n_clases)

    def forward(self, input_ids, attention_mask):
        # Devuelve los logits (reviews x n_clases)
        return self.bert(input_ids=input_ids, attention_mask=attention_mask).logits


class BertClassifier(ClassifierMixin, BaseEstimator):
    """
    Ajuste fino de BERT: se entrenan todos sus pesos (no solo la capa de salida) con nuestras
    reviews. Recibe los textos tal cual: el tokenizador de BERT los parte en subpalabras.
    """

    def __init__(self, nombre='prajjwal1/bert-tiny', longitud=128):
        self.nombre = nombre        # modelo de Hugging Face (bert-tiny: 2 capas, 4,4 M de parámetros)
        self.longitud = longitud    # subpalabras por review

    def _tokenizar(self, textos):
        """Textos -> diccionario con input_ids y attention_mask (tensores reviews x longitud)."""
        t = self.tokenizador_(list(textos), max_length=self.longitud, padding='max_length',
                              truncation=True, return_tensors='pt')
        return {'input_ids': t['input_ids'], 'attention_mask': t['attention_mask']}

    def fit(self, X, y, X_val, y_val):
        """
        Ajusta BERT. X y X_val son listas de textos (str).
        Se para cuando la pérdida en validación deja de bajar (early stopping).
        """
        self.tokenizador_ = AutoTokenizer.from_pretrained(self.nombre)
        self.classes_ = np.unique(y)
        y = np.searchsorted(self.classes_, y)
        y_val = np.searchsorted(self.classes_, y_val)
        pesos = compute_class_weight('balanced', classes=np.unique(y), y=y)

        self.modelo_ = NeuralNetClassifier(
            RedBert,
            module__nombre=self.nombre,
            module__n_clases=len(self.classes_),
            criterion=torch.nn.CrossEntropyLoss,
            criterion__weight=torch.tensor(pesos, dtype=torch.float32),
            optimizer=torch.optim.AdamW,
            lr=0.0001,              # pequeña: los pesos de BERT ya vienen entrenados
            max_epochs=5,
            batch_size=32,
            iterator_train__shuffle=True,
            train_split=predefined_split(Dataset(self._tokenizar(X_val), y_val)),
            callbacks=[EarlyStopping(patience=2, load_best=True)],
            device=DISPOSITIVO,
        )
        self.modelo_.fit(self._tokenizar(X), y)
        return self

    def predict(self, X):
        """Devuelve la clase predicha ('negativas', 'neutras' o 'positivas') de cada review."""
        return self.classes_[self.modelo_.predict(self._tokenizar(X))]
