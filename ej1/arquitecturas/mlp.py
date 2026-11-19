"""
mlp.py - Perceptrón multicapa sobre el embedding de documento (media de los embeddings de palabra).

    índices -> Embedding (GloVe) -> media de las palabras -> Linear + ReLU + Dropout -> Linear (logits)
"""
import torch
from torch import nn


class MLP(nn.Module):

    def __init__(self, matriz_embeddings, n_clases, entrenable=False, neuronas=128, dropout=0.3):
        super().__init__()
        # TODO:
        #   self.embedding = nn.Embedding.from_pretrained(torch.tensor(matriz_embeddings),
        #                                                 freeze=not entrenable, padding_idx=0)
        #   self.oculta = nn.Linear(dimensión de los embeddings, neuronas)
        #   self.dropout = nn.Dropout(dropout)
        #   self.salida = nn.Linear(neuronas, n_clases)
        raise NotImplementedError

    def forward(self, X):
        # X: (reviews x longitud) con los índices de las palabras; 0 = relleno
        # TODO:
        #   1. Embeddings de palabra:  E = self.embedding(X)          -> (reviews x longitud x dim)
        #   2. Embedding de documento: media de E sin contar el relleno
        #        mascara = (X != 0).unsqueeze(-1).float()
        #        doc = (E * mascara).sum(dim=1) / mascara.sum(dim=1).clamp(min=1)
        #   3. Capa oculta con ReLU y dropout, y capa de salida
        #   4. Devolver los logits (sin softmax: CrossEntropyLoss ya lo aplica)
        raise NotImplementedError
