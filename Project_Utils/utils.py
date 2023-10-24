import numpy as np
import pandas as pd
from PIL import Image
import geocoder

def arrToPIL(image: np.ndarray):
    imageOBJ = Image.fromarray(image)
    return imageOBJ

def getGrayColorImg(colorArr: np.ndarray):
    gray_arr = np.asarray(Image.fromarray(colorArr).convert("L"))
    return [colorArr, gray_arr]

def getCurrentLocation():
    g = geocoder.ip('me')
    return pd.DataFrame([g.latlng], columns=['LAT', 'lon'])