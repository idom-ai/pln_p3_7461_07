"""
pln_p3_7461_07_e2.py - EJERCICIO 2: ajuste fino de BERT (bert-tiny) en los dos corpus.

  - Mismas reviews, folds y validación que en el ejercicio 1 (50000 reviews, 3 folds, 10 % del
    train para validación, semilla 42).
  - BERT recibe el texto de cada review (ya limpio por el reader, sin quitar stopwords).
  - Muestra el F1 macro en test (media de los 3 folds) y guarda en resultados/resultados_e2.json
    las pérdidas de cada época y las predicciones en test (para visualizar_resultados.py).

Antes de ejecutarlo hay que descomprimir los .zip de CORPUS dentro de data/ (en la raíz).
La primera vez se descarga bert-tiny de Hugging Face (~17 MB).
"""
import json
import os
import sys

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# Carpeta raíz de la práctica (la de arriba): ahí están P2/, data/ y resultados/
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)                          # para poder importar P2/
sys.path.insert(0, os.path.join(RAIZ, 'ej1'))     # y torch_classifier.py

from torch_classifier import BertClassifier
from P2.corpus_loader import CorpusLoader
from P2.json_corpus_reader import JsonCorpusReader

CORPUS = {'Gift_Cards': os.path.join(RAIZ, 'data', 'corpus_pos54_neu32'),
          'All_Beauty': os.path.join(RAIZ, 'data', 'corpus_All_Beauty_pos54_neu32')}
RESULTADOS = os.path.join(RAIZ, 'resultados')
N_REVIEWS = 50000

resultados = []
for corpus, carpeta in CORPUS.items():
    reader = JsonCorpusReader(carpeta)
    loader = CorpusLoader(reader, folds=3, n_total=N_REVIEWS)
    etiquetas = np.array(loader.labels())
    textos = np.array(list(reader.texts(fileids=list(loader.fileids()))))
    folds = loader.splitter.split(loader.fileids(), etiquetas)

    for fold, (train, test) in enumerate(folds, start=1):
        train, val = train_test_split(train, test_size=0.1, stratify=etiquetas[train], random_state=42)
        print(f'\n{corpus} - fold {fold} - BERT')
        clf = BertClassifier()
        clf.fit(textos[train], etiquetas[train], textos[val], etiquetas[val])
        pred = clf.predict(textos[test])
        print(f'>> F1 macro en test {f1_score(etiquetas[test], pred, average="macro"):.3f}')
        resultados.append({
            'corpus': corpus,
            'modelo': 'BERT-tiny',
            'fold': fold,
            'train_loss': list(clf.modelo_.history[:, 'train_loss']),
            'valid_loss': list(clf.modelo_.history[:, 'valid_loss']),
            'y_test': list(etiquetas[test]),
            'y_pred': list(pred),
        })

os.makedirs(RESULTADOS, exist_ok=True)
with open(os.path.join(RESULTADOS, 'resultados_e2.json'), 'w') as f:
    json.dump(resultados, f)

print('\n===== F1 MACRO EN TEST (media de los 3 folds) =====')
for corpus in CORPUS:
    f1 = [f1_score(r['y_test'], r['y_pred'], average='macro') for r in resultados if r['corpus'] == corpus]
    print(f'  {corpus:12s} BERT-tiny {np.mean(f1):.3f}')
print('\nResultados guardados en resultados/resultados_e2.json')
