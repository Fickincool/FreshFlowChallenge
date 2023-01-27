## Requirements

Create the environment with all dependencies:

`conda create --name <env> --file requirements.txt`

Install the utilities module

`pip install -e FFlowUtils `

# Summary

I first did and EDA and discovered that there where douplicated entries, also that two of the items have small availability at some periods of the year, probably due to harvesting seasons. I resampled the data by item per der because this was the timeliness required for the task

I tried to train several models and the idea was that in the end I would take an average of their predictions. This is usually a good idea because different models might catch different aspects of the data and the ensemble is likely to be less noisy and yield better predictions if individual errors are uncorrelated.

The train.py file trains two of these models, an ARIMA model and an XGBoost model.