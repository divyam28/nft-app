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
from sklearn.metrics import mean_squared_error
import math
import PIL
from tqdm import tqdm
import pandas as pd
import re
from keybert import KeyBERT
from PIL import Image
PIL.Image.MAX_IMAGE_PIXELS = None

np.random.seed(seed=0)

from utils import *

st.set_page_config(page_title='Price Range Recommendation for NFT',layout='wide')

st.title('Price Range Recommendation for NFT')
st.text('Get a good estimation of how much your art is worth!')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_image(image_file):
	img = Image.open(image_file)
	return img

final_model = NFTModelA()
final_model.load_state_dict(torch.load('model_2065.713518857956_31'))
final_model.eval()

st.subheader("Upload your art here!")

# st.image(load_image(image_file),width=250)
image_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

k = pd.read_csv('../nft_sales.csv')


if (image_file is not None):
        # To View Uploaded Image
        st.image(load_image(image_file),width=200)
        img = get_img_features(load_image(image_file))

        description = st.text_input('Describe it!')
        est = st.button('Estimate')
        if est:
                txt = rem_emo(description)
                txt = keyword_extract(txt)
                emb = embedding_dim(get_embedding(txt))

                img = torch.tensor(img).unsqueeze(0)
                emb = torch.tensor(emb).unsqueeze(0)

                price = denormalize(final_model(img,emb).detach().item())
                
                st.subheader(f"You should enter the market between  {round(price,3)} ETH - {round(price+np.random.random(),3)} ETH")

                # asset_id = image_file.name[:-4]
                # actual = k[k['asset_id'] == int(asset_id)]['event_total_price'].mean()

                # st.subheader(f"Actual price is {actual}")
                

# get_img_features(load_image(image_file))



# st.subheader('You should price it between 0.003 - 0.008 ETH.')