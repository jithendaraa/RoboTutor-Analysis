import pandas as pd
import numpy as np

df = pd.read_csv("Data/village_114.txt", delimiter='\t')[:10]
df.to_excel("Data/village114.xlsx")