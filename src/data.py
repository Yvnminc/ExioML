import pandas as pd
from sklearn.model_selection import train_test_split

# Read in the data
type = "pxp"

df = pd.read_csv(f'../data/{type}/{type}_clean_full.csv')
df.head()

def get_train_vail_test(type = "pxp", data = "clean", ratio = 0.2):
    df = pd.read_csv(f'../data/{type}/{type}_{data}_full.csv')
    
    train, test = train_test_split(df, test_size=ratio, random_state=42)
    train, val = train_test_split(train, test_size=ratio)
    
    return train, val, test