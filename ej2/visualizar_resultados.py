"""
visualizar_resultados.py - Gráficas del rendimiento de los modelos de los ejercicios 1 y 2.

Lee resultados/resultados_e1.json y resultados/resultados_e2.json (los que existan; resultados/
está en la raíz de la práctica) y guarda en resultados/, para cada corpus:
  f1_macro_<corpus>.png            F1 macro en test de cada modelo (media y desviación de los
                                   3 folds), junto al mejor modelo de la P2
  f1_por_clase_<corpus>.png        F1 en test de cada clase y modelo
  accuracy_por_clase_<corpus>.png  accuracy de cada clase (recall: % de reviews de la clase que
                                   se aciertan) y modelo, junto a la P2
  curvas_loss_<corpus>.png         pérdida de entrenamiento y de validación por época (fold 1)
  matrices_confusion_<corpus>.png  matriz de confusión de cada modelo (suma de los 3 folds, en %
                                   de cada clase real)
"""
import json
import os

import matplotlib
matplotlib.use('Agg')   # dibuja en fichero, sin abrir ventanas
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import confusion_matrix, f1_score, recall_score

CLASES = ['negativas', 'neutras', 'positivas']
CARPETA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'resultados')

# Mejor modelo de la P2 en cada corpus (test, media de 3 folds): F1 macro y accuracy por clase
P2 = {'Gift_Cards': ('P2: SGDClassifier', 0.635, [0.79, 0.39, 0.92]),
      'All_Beauty': ('P2: LogisticRegression', 0.706, [0.73, 0.59, 0.88])}

# Un color fijo por modelo (paleta categórica validada) y gris para la P2
COLORES = {'MLP + GloVe': '#2a78d6', 'RNN + GloVe': '#eb6834', 'MLP + Word2Vec': '#1baf7a',
           'RNN + Word2Vec': '#eda100', 'BERT-tiny': '#e87ba4'}
GRIS = '#a3a29c'
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#b5b4ad', 'axes.labelcolor': '#52514e',
                     'xtick.color': '#52514e', 'ytick.color': '#52514e', 'font.size': 10})

# Resultados de los dos ejercicios
resultados = []
for fichero in ['resultados_e1.json', 'resultados_e2.json']:
    ruta = os.path.join(CARPETA, fichero)
    if os.path.exists(ruta):
        with open(ruta) as f:
            resultados += json.load(f)


def modelos_de(corpus):
    """Modelos evaluados en el corpus, en el orden de COLORES."""
    presentes = {r['modelo'] for r in resultados if r['corpus'] == corpus}
    return [m for m in COLORES if m in presentes]


def de(corpus, modelo):
    """Resultados (uno por fold) de un modelo en un corpus."""
    return [r for r in resultados if r['corpus'] == corpus and r['modelo'] == modelo]


def guardar(fig, nombre):
    fig.savefig(os.path.join(CARPETA, nombre), dpi=150)
    plt.close(fig)
    print(f'Guardada resultados/{nombre}')


def grafica_f1_macro(corpus):
    """Barras horizontales: F1 macro de cada modelo (media ± desviación de los folds) y la P2."""
    modelos = modelos_de(corpus)
    medias = [np.mean([f1_score(r['y_test'], r['y_pred'], average='macro') for r in de(corpus, m)])
              for m in modelos]
    desviaciones = [np.std([f1_score(r['y_test'], r['y_pred'], average='macro') for r in de(corpus, m)])
                    for m in modelos]
    nombre_p2, f1_p2, _ = P2[corpus]
    etiquetas = modelos + [nombre_p2]
    valores = medias + [f1_p2]
    colores = [COLORES[m] for m in modelos] + [GRIS]

    fig, ax = plt.subplots(figsize=(7, 0.55 * len(etiquetas) + 1), layout='constrained')
    y = np.arange(len(etiquetas))[::-1]
    ax.barh(y, valores, color=colores, height=0.6, xerr=desviaciones + [0],
            error_kw={'ecolor': '#52514e', 'capsize': 3, 'lw': 1})
    for yi, v, d in zip(y, valores, desviaciones + [0]):
        ax.text(v + d + 0.015, yi, f'{v:.3f}', va='center', color='#0b0b0b')   # tras la barra de error
    ax.set_yticks(y, etiquetas)
    ax.set_xlim(0, 1)
    ax.set_xlabel('F1 macro en test (media de 3 folds)')
    ax.set_title(f'F1 macro por modelo - {corpus}', loc='left')
    ax.grid(axis='x', color='#e1e0d9', lw=0.8)
    ax.set_axisbelow(True)
    guardar(fig, f'f1_macro_{corpus}.png')


def grafica_por_clase(corpus, metrica):
    """Barras agrupadas por clase: F1 o accuracy (recall) de cada clase para cada modelo."""
    modelos = modelos_de(corpus)
    valores = {}
    for m in modelos:
        por_fold = []
        for r in de(corpus, m):
            if metrica == 'f1':
                por_fold.append(f1_score(r['y_test'], r['y_pred'], labels=CLASES, average=None))
            else:
                por_fold.append(recall_score(r['y_test'], r['y_pred'], labels=CLASES, average=None))
        valores[m] = np.mean(por_fold, axis=0)
    colores = [COLORES[m] for m in modelos]
    if metrica == 'accuracy':   # de la P2 solo hay accuracy por clase
        nombre_p2, _, accuracy_p2 = P2[corpus]
        valores[nombre_p2] = accuracy_p2
        colores.append(GRIS)

    fig, ax = plt.subplots(figsize=(9, 4.5), layout='constrained')
    ancho = 0.8 / len(valores)
    x = np.arange(len(CLASES))
    for i, (m, v) in enumerate(valores.items()):
        posiciones = x - 0.4 + ancho * (i + 0.5)
        ax.bar(posiciones, v, width=ancho * 0.9, color=colores[i], label=m)
        for p, vi in zip(posiciones, v):
            ax.text(p, vi + 0.01, f'{vi:.2f}', ha='center', fontsize=7, color='#52514e')
    ax.set_xticks(x, CLASES)
    ax.set_ylim(0, 1.05)
    nombre = 'F1' if metrica == 'f1' else 'Accuracy (recall)'
    ax.set_ylabel(f'{nombre} en test (media de 3 folds)')
    ax.set_title(f'{nombre} por clase - {corpus}', loc='left')
    ax.grid(axis='y', color='#e1e0d9', lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc='upper left')
    guardar(fig, f'{metrica}_por_clase_{corpus}.png')


def grafica_loss(corpus):
    """Un panel por modelo: pérdida de entrenamiento y de validación en cada época (fold 1)."""
    modelos = modelos_de(corpus)
    fig, ejes = plt.subplots(1, len(modelos), figsize=(3.2 * len(modelos), 3.4), squeeze=False,
                             layout='constrained')
    for ax, m in zip(ejes[0], modelos):
        r = de(corpus, m)[0]
        epocas = np.arange(1, len(r['train_loss']) + 1)
        ax.plot(epocas, r['train_loss'], color='#2a78d6', lw=2, marker='o', ms=4, label='entrenamiento')
        ax.plot(epocas, r['valid_loss'], color='#eb6834', lw=2, marker='o', ms=4, label='validación')
        mejor = int(np.argmin(r['valid_loss']))     # la época que se queda el early stopping
        ax.axvline(epocas[mejor], color=GRIS, ls='--', lw=1)
        ax.set_title(m, fontsize=10)
        ax.set_xlabel('época')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))   # épocas enteras
        ax.ticklabel_format(axis='y', useOffset=False)
        ax.grid(color='#e1e0d9', lw=0.8)
    ejes[0][0].set_ylabel('pérdida (entropía cruzada)')
    ejes[0][0].legend(frameon=False, fontsize=8)
    fig.suptitle(f'Curvas de pérdida (fold 1; línea discontinua = mejor época) - {corpus}',
                 x=0.01, ha='left')
    guardar(fig, f'curvas_loss_{corpus}.png')


def grafica_confusion(corpus):
    """Una matriz de confusión por modelo, sumando los folds y en % de cada clase real."""
    modelos = modelos_de(corpus)
    fig, ejes = plt.subplots(1, len(modelos), figsize=(3.2 * len(modelos), 3.8), squeeze=False,
                             layout='constrained')
    for ax, m in zip(ejes[0], modelos):
        cm = sum(confusion_matrix(r['y_test'], r['y_pred'], labels=CLASES) for r in de(corpus, m))
        cm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(cm, cmap='Blues', vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{cm[i, j]:.0%}', ha='center', va='center',
                        color='white' if cm[i, j] > 0.5 else '#0b0b0b')
        ax.set_xticks(range(3), ['neg', 'neu', 'pos'])
        ax.set_yticks(range(3), ['neg', 'neu', 'pos'])
        ax.set_xlabel('predicha')
        ax.set_title(m, fontsize=10)
        ax.spines[:].set_visible(False)
    ejes[0][0].set_ylabel('real')
    fig.suptitle(f'Matrices de confusión (% de cada clase real, 3 folds) - {corpus}', x=0.01, ha='left')
    guardar(fig, f'matrices_confusion_{corpus}.png')


for corpus in sorted({r['corpus'] for r in resultados}):
    grafica_f1_macro(corpus)
    grafica_por_clase(corpus, 'f1')
    grafica_por_clase(corpus, 'accuracy')
    grafica_loss(corpus)
    grafica_confusion(corpus)
