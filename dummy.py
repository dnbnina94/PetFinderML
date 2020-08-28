import pandas as pd
import numpy as np
import math
import os

df = pd.read_csv("test_img_features.csv")
df = df.rename(columns=lambda x: "Image Feature " + x)

if os.path.exists("test_img_features.csv"):
  os.remove("test_img_features.csv")
df.to_csv("test_img_features.csv")