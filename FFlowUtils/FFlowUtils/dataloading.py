import pandas as pd
import os

dateparse = lambda x: pd.to_datetime(x, errors='coerce', format='%Y-%m-%d')

DATA_FOLDER = '/home/jeronimo/Desktop/Freshflow_techChallenge/FreshFlowChallenge/data/'

def load_sales():

    sales = pd.read_csv(os.path.join(DATA_FOLDER, 'data.csv'), parse_dates=['day'], date_parser=dateparse).drop('Unnamed: 0', axis=1)
    sales.drop_duplicates(inplace=True)
    print(sales.shape)
    
    return sales