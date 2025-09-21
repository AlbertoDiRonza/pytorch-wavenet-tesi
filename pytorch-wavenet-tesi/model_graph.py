# VISUALIZZAZIONE DEL GRAFO DEL MODELLO
import torch
import torch.nn as nn
from wavenet_model import *
from visualize import make_dot

dtype = torch.FloatTensor
# PUò AVERE SENSO GRAFICARE QUALCOSA DI PIù RIDOTTO? 
model = WaveNetModel(layers=1,
                    blocks=1,
                    dilation_channels=2,
                    residual_channels=2,
                    skip_channels=16,
                    end_channels=8,
                    output_length=1,
                    dtype=dtype,
                    bias=True)

# DEVE AVERE LA SHAPE E TIPO DEI DATI CON CUI LAVORA IL MODELLO, QUI NON STO FACENDO TRAINING PER IL GRAFO IL TARGET NON SERVE A NULLA
dummy_input = torch.randn(1, 256, 16000)
output = model(dummy_input)

dot = make_dot(output, dict(model.named_parameters()))
dot.render("model_graph_2", format="svg")