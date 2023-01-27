#!/usr/bin/env python3

import os
import sys
import argparse

from FFlowUtils.dataloading import load_sales, DATA_FOLDER

import pandas as pd
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl

from pmdarima import auto_arima

def preprocess():
    sales = load_sales()
    sales = sales.set_index('day').groupby('item_name').apply(lambda x: x.resample('d').sum())
    sales.reset_index(inplace=True)
    # sales['split'] = np.where(sales.day>'2021-12-31', 'validation', 'train')

    return sales

def autoArima_item_model(item_sales, seasonal):
    weekly_model = auto_arima(item_sales.set_index('day').sales_quantity,
                    start_p=1,
                    start_q=1,
                    max_p=3,
                    max_q=3, 
                    m=7,
                    start_P=0,
                    seasonal=seasonal,
                    d=None,
                    D=None,
                    trace=True,
                    error_action='trace',  
                    suppress_warnings=True, 
                    stepwise=True)


    return weekly_model

def main(arguments):

    # parser = argparse.ArgumentParser(
    #     description=__doc__,
    #     formatter_class=argparse.RawDescriptionHelpFormatter)
    # parser.add_argument('infile', help="Input file", type=argparse.FileType('r'))
    # parser.add_argument('-o', '--outfile', help="Output file",
    #                     default=sys.stdout, type=argparse.FileType('w'))

    # args = parser.parse_args(arguments)

    sales = preprocess()
    
    item_list = list(sales.item_name.unique())

    arima_dict = {}
    for item in item_list:
        item_sales = sales[sales.item_name==item]
        # we won't use seasonality for Tangerines
        if item=='SL MANDARINEN BEH.ES I 750G GS':
            seasonal = False
        else:
            seasonal = True

        model = autoArima_item_model(item_sales, seasonal)
        
        arima_dict[item] = model

    # Store data (serialize)
    arima_model_file = os.path.join(DATA_FOLDER, 'arima_model_by_item.pkl')
    with open(arima_model_file, 'wb') as handle:
        pkl.dump(arima_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))