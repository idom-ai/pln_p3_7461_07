"""
rnn.py - Red recurrente LSTM (opcionalmente bidireccional).

    índices -> Embedding (GloVe) -> LSTM -> último estado oculto -> Dropout -> Linear (logits)

El embedding de documento es el último estado oculto de la LSTM, que lee las palabras en orden.
"""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class RNN(nn.Module):

    def __init__(self, matriz_embeddings, n_clases, entrenable=False, bidireccional=False,
                 unidades=64, dropout=0.3):
        super().__init__()
        self.bidireccional = bidireccional
        # TODO:
        #   self.embedding = nn.Embedding.from_pretrained(torch.tensor(matriz_embeddings),
        #                                                 freeze=not entrenable, padding_idx=0)
        #   self.lstm = nn.LSTM(dimensión de los embeddings, unidades,
        #                       batch_first=True, bidirectional=bidireccional)
        #   self.dropout = nn.Dropout(dropout)
        #   self.salida = nn.Linear(unidades (x2 si es bidireccional), n_clases)
        raise NotImplementedError

    def forward(self, X):
        # X: (reviews x longitud) con los índices de las palabras; 0 = relleno (siempre al final)
        # TODO:
        #   1. E = self.embedding(X)
        #   2. Para que la LSTM no lea el relleno:
        #        longitudes = (X != 0).sum(dim=1).clamp(min=1).cpu()
        #        E = pack_padded_sequence(E, longitudes, batch_first=True, enforce_sorted=False)
        #   3. _, (h, _) = self.lstm(E)
        #   4. Último estado oculto: h[-1]; si es bidireccional, torch.cat([h[-2], h[-1]], dim=1)
        #   5. Dropout y capa de salida; devolver los logits
        raise NotImplementedError
