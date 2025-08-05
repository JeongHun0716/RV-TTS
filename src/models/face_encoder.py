import torch.nn as nn
import sys
sys.path.append('./insightface/recognition/arcface_torch')
from backbones import get_model
import torch.nn.functional as F
import torch
import numpy as np

class Face_Encoder(nn.Module):
    def __init__(self, name='r50'):
        super().__init__()
        self.model = get_model(name)
        self.embedding = nn.Linear(512, 256)
        self.audio_embedding = nn.Linear(192, 256)
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = nn.Parameter(torch.ones([]) * np.log(10))
        self.temp_bias = nn.Parameter(torch.ones([]) * -10)

    def forward(self, image, audio_embedding):
        embedding = self.embedding(self.model(image))
        audio_embedding = self.audio_embedding(audio_embedding)

        face_embed = F.normalize(embedding)    #B, 192
        audio_embed = F.normalize(audio_embedding)  #B, 192

        similarity = face_embed @ audio_embed.t() * self.temperature.exp() + self.temp_bias  # B, B
        loss = 0.5 * (self.criterion(similarity, torch.arange(similarity.size(0), device=similarity.device)) 
                        + self.criterion(similarity.t(), torch.arange(similarity.size(0), device=similarity.device)))

        return embedding, similarity, loss
    
    def forward_face(self, image):
        embedding = self.embedding(self.model(image))
        return embedding

    def forward_audio(self, audio):
        embedding = self.audio_embedding(audio)
        return embedding