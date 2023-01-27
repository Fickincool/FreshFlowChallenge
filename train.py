#!/usr/bin/env python3

import os
import sys
import argparse

from FFlowUtils.dataloading import load_sales, DATA_FOLDER
from FFlowUtils.modeling  import autoArima_item_model, regression_model

from xgboost import XGBRegressor

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

def main(arguments):

    # we always use the data in sales to make predictions
    sales = preprocess()
    
    item_list = list(sales.item_name.unique())

    arima_dict = {}
    regressor_dict = {}
    for item in item_list:
        item_sales = sales[sales.item_name==item]
        # we won't use seasonality for Tangerines
        if item=='SL MANDARINEN BEH.ES I 750G GS':
            seasonal = False
        else:
            seasonal = True

        arima_model = autoArima_item_model(item_sales, seasonal)

        regressor = XGBRegressor()
        regression_model(regressor, item_sales)

        arima_dict[item] = arima_model
        regressor_dict[item] = regressor

    # Store data (serialize)
    arima_model_file = os.path.join(DATA_FOLDER, 'arima_model_by_item.pkl')
    with open(arima_model_file, 'wb') as handle:
        pkl.dump(arima_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)

    # Store data (serialize)
    regressor_model_file = os.path.join(DATA_FOLDER, 'regressor_model_by_item.pkl')
    with open(regressor_model_file, 'wb') as handle:
        pkl.dump(regressor_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))