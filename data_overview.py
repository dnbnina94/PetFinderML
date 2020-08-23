import pandas as pd
from pandasgui import show
import math

train = pd.read_csv("train.csv");

print(train.info())

# # show(breeds, settings={'block': True});
# show(train, settings={'block': True});
# show(train.describe(), settings={'block': True});