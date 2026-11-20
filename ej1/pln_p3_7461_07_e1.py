"""
pln_p3_7461_07_e1.py - EJERCICIO 1: MLP y RNN con embeddings estáticos (GloVe y Word2Vec), en dos corpus.

  1. Mismas reviews y mismos 3 folds que en la P2 (50000 reviews, semilla 42).
  2. En cada fold, el 10 % del train se separa como validación.
  3. De cada modelo se prueban las configuraciones de CONFIGURACIONES; se elige la de mejor
     F1 macro en validación y se mide en el test del fold.
  4. Muestra el F1 macro en test (media de los 3 folds) y guarda en resultados/resultados_e1.json
     las pérdidas de cada época y las predicciones en test (para visualizar_resultados.py).

Los corpus se descomprimen solos (de data/*.zip a corpus/) la primera vez que se ejecuta.
La primera vez se descargan los embeddings (~400 MB GloVe y ~1,6 GB Word2Vec) en ~/gensim-data.
"""
import json
import os
import sys
import zipfile

import gensim.downloader
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# Carpeta raíz de la práctica (la de arriba): ahí están P2/, classifiers.py, data/, corpus/ y resultados/
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)   # para poder importar P2/ y classifiers.py

from arquitecturas.mlp import MLP
from arquitecturas.rnn import RNN
from embeddings import Vectorizador, matriz_embeddings, tokenizar
from classifiers import EmbeddingClassifier
from P2.corpus_loader import CorpusLoader
from P2.json_corpus_reader import JsonCorpusReader

# Los corpus están comprimidos en data/ (versionados con DVC). La primera vez se descomprimen en
# corpus/, fuera de data/, para que DVC no los vea como un cambio de los datos.
CARPETA_CORPUS = os.path.join(RAIZ, 'corpus')
ZIPS = {'Gift_Cards': 'corpus_pos54_neu32', 'All_Beauty': 'corpus_All_Beauty_pos54_neu32'}
for nombre in ZIPS.values():
    if not os.path.isdir(os.path.join(CARPETA_CORPUS, nombre)):
        print(f'Descomprimiendo data/{nombre}.zip en corpus/...')
        with zipfile.ZipFile(os.path.join(RAIZ, 'data', nombre + '.zip')) as z:
            z.extractall(CARPETA_CORPUS)
CORPUS = {corpus: os.path.join(CARPETA_CORPUS, nombre) for corpus, nombre in ZIPS.items()}
RESULTADOS = os.path.join(RAIZ, 'resultados')
EMBEDDINGS = {'GloVe': 'glove-wiki-gigaword-300',        # los dos de 300 dimensiones
              'Word2Vec': 'word2vec-google-news-300'}
N_REVIEWS = 50000

CONFIGURACIONES = {
    'MLP': (MLP, [{'entrenable': False}, {'entrenable': True}]),
    'RNN': (RNN, [{'bidireccional': False}, {'bidireccional': True}]),
}


def evaluar(corpus, embeddings, textos, etiquetas, folds, vectores):
    """
    Entrena y evalúa cada modelo en los 3 folds de un corpus con unos embeddings.
    Devuelve una lista con un resultado por modelo y fold.
    """
    resultados = []
    for fold, (train, test) in enumerate(folds, start=1):
        train, val = train_test_split(train, test_size=0.1, stratify=etiquetas[train], random_state=42)
        vectorizador = Vectorizador().fit([textos[i] for i in train])
        matriz = matriz_embeddings(vectorizador, vectores)
        X_train = vectorizador.transform([textos[i] for i in train])
        X_val = vectorizador.transform([textos[i] for i in val])
        X_test = vectorizador.transform([textos[i] for i in test])

        for modelo, (red, configuraciones) in CONFIGURACIONES.items():
            # La mejor configuración en validación...
            mejor_f1 = -1
            for parametros in configuraciones:
                print(f'\n{corpus} - {embeddings} - fold {fold} - {modelo} {parametros}')
                clf = EmbeddingClassifier(red, matriz, parametros)
                clf.fit(X_train, etiquetas[train], X_val, etiquetas[val])
                f1_val = f1_score(etiquetas[val], clf.predict(X_val), average='macro')
                print(f'F1 macro en validación: {f1_val:.3f}')
                if f1_val > mejor_f1:
                    mejor_f1, mejor_clf, mejores_parametros = f1_val, clf, parametros

            # ...medida en test
            pred = mejor_clf.predict(X_test)
            print(f'>> {modelo} {mejores_parametros}: F1 macro en test '
                  f'{f1_score(etiquetas[test], pred, average="macro"):.3f}')
            resultados.append({
                'corpus': corpus,
                'modelo': f'{modelo} + {embeddings}',
                'fold': fold,
                'parametros': str(mejores_parametros),
                'train_loss': list(mejor_clf.modelo_.history[:, 'train_loss']),
                'valid_loss': list(mejor_clf.modelo_.history[:, 'valid_loss']),
                'y_test': list(etiquetas[test]),
                'y_pred': list(pred),
            })
    return resultados


# 1. Reviews y folds de cada corpus (se tokenizan una sola vez)
datos = {}
for corpus, carpeta in CORPUS.items():
    print(f'Tokenizando {corpus}...')
    reader = JsonCorpusReader(carpeta)
    loader = CorpusLoader(reader, folds=3, n_total=N_REVIEWS)
    etiquetas = np.array(loader.labels())
    folds = list(loader.splitter.split(loader.fileids(), etiquetas))
    datos[corpus] = (tokenizar(reader, loader.fileids()), etiquetas, folds)

# 2. Cada embeddings con cada corpus
resultados = []
for embeddings, nombre in EMBEDDINGS.items():
    print(f'\nCargando {nombre}...')
    vectores = gensim.downloader.load(nombre)
    for corpus in CORPUS:
        resultados += evaluar(corpus, embeddings, *datos[corpus], vectores)

os.makedirs(RESULTADOS, exist_ok=True)
with open(os.path.join(RESULTADOS, 'resultados_e1.json'), 'w') as f:
    json.dump(resultados, f)

# 3. Resumen: F1 macro en test, media de los 3 folds
print('\n===== F1 MACRO EN TEST (media de los 3 folds) =====')
for corpus in CORPUS:
    print(f'\n{corpus}')
    for modelo in sorted({r['modelo'] for r in resultados}):
        f1 = [f1_score(r['y_test'], r['y_pred'], average='macro')
              for r in resultados if r['corpus'] == corpus and r['modelo'] == modelo]
        print(f'  {modelo:20s} {np.mean(f1):.3f}')
print('\nResultados guardados en resultados/resultados_e1.json')
