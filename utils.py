import streamlit as st
import pandas as pd
import numpy as np
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import time
import torch
import os
import time
import re
from sklearn.metrics import mean_squared_error
import math
import PIL
from tqdm import tqdm
from PIL import Image
from keybert import KeyBERT
PIL.Image.MAX_IMAGE_PIXELS = None


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
vgg = models.vgg16(pretrained=True).to(device)
kw_model = KeyBERT()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bertmodel = BertModel.from_pretrained("bert-base-uncased")
bertmodel.to(device)

def get_img_features(img):
    try:
        img = img.convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ])
        imgt = transform(img).to(device)
        features = vgg.features(imgt.unsqueeze(0)).squeeze(0).to('cpu').detach().numpy()
    except PIL.UnidentifiedImageError:
        return -1
    
    return features

def rem_emo(string):
    temp=[]
    for j in (string.split()):
        s = re.sub(r'[^a-zA-Z]', '', j)
        temp.append(s)
    new_desc = re.sub(' +',' ', ' '.join(temp))
    
    return new_desc

def keyword_extract(s):
    if s == '':
        return ''
    elif len(s.split()) < 12:
        f = kw_model.extract_keywords(s,keyphrase_ngram_range=(0, 3), stop_words=None)[0][0]
    else:
        f = kw_model.extract_keywords(s,keyphrase_ngram_range=(1, 10), stop_words=None)[0][0]
    return f


class NFTModelA(torch.nn.Module):

    def __init__(self):
        super(NFTModelA, self).__init__()
        
        self.first =  torch.nn.Sequential(
            torch.nn.Conv2d(512,512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512,512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            torch.nn.AdaptiveAvgPool2d((7,7))
        )
        
        self.conv_part = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(25088,4096,bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.1,inplace=False),
            torch.nn.Linear(4096,4096)
        )
        
        self.txt_part = torch.nn.Sequential(
            torch.nn.Linear(768, 768)
        )
        
        self.final = torch.nn.Sequential(
            torch.nn.Linear(768+4096, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.1, inplace=False),
            torch.nn.Linear(1024,256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.1, inplace=False),
            torch.nn.Linear(256,1),
            torch.nn.Sigmoid()
        )

    def forward(self, img, emb):
        x = self.first(img)
        con = self.conv_part(x)
        emb = self.txt_part(emb)
        merge = torch.cat((con,emb),1)
        x = self.final(merge)
        return x


def get_embedding(t):
    enc = tokenizer(t, return_tensors='pt').to(device)
    output = bertmodel(**enc)
    
    return output[0].to('cpu').detach().numpy()

def embedding_dim(x):
    return np.mean(x.squeeze(), axis=0)


def denormalize(x):
    xmin = 0.0
    xmax = 9.99999
    diff = xmax - xmin
    
    return x*diff + xmin