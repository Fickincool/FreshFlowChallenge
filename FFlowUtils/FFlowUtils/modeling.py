
from pmdarima import auto_arima
from xgboost import XGBRegressor

from FFlowUtils.features import backshift_df, add_date_features


def autoArima_item_model(item_sales, seasonal):
    """
    This function takes in a dataframe of sales data for a single item, and a boolean value for whether
    or not the data is seasonal. It then fits an ARIMA model to the data, and returns the model.
    
    Args:
      item_sales: the dataframe of the item sales
      seasonal: If True, the model is fit using SARIMAX(p,d,q)(P,D,Q,s)
    
    Returns:
      The model is being returned.
    """
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

def regression_model(regressor, item_sales):
    """
    > We're going to take the dataframe of item sales, shift the sales quantity column back by 1, 2, 3,
    4, 5, 6, 7, and 8 days, add the date features, and then fit the regressor to the data
    
    Args:
      regressor: the regressor you want to use.
      item_sales: the dataframe containing the sales data
    
    Returns:
      The regressor object
    """
    item_sales = backshift_df(item_sales, ['sales_quantity'], 1, 8)
    item_sales = add_date_features(item_sales, 'day')

    feature_list = ['sales_quantity_1',
       'sales_quantity_2', 'sales_quantity_3', 'sales_quantity_4',
       'sales_quantity_5', 'sales_quantity_6', 'sales_quantity_7',
       'sales_quantity_8', 'year', 'month', 'day_number', 'dayofyear',
       'dayofweek', 'weekofyear', 'is_weekend']
    
    dep_var = 'sales_quantity'

    X = item_sales[feature_list]
    y = item_sales[dep_var]

    regressor.fit(X, y)

    return regressor