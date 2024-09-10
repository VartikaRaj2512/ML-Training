# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 12:14:22 2024

@author: DELL
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.graphics.tsaplots as tsa_plots
from statsmodels.tsa.ar_model import AutoReg

# Load the data
Walmart = pd.read_csv(r"C:\Users\DELL\Desktop\Github\ML-Training\Walmart Footfalls Raw.csv")

# Pre-processing
Walmart["t"] = np.arange(1, 160)
Walmart["t_square"] = Walmart["t"] ** 2
Walmart["log_footfalls"] = np.log(Walmart["Footfalls"])

# Extract month abbreviations
Walmart['months'] = Walmart["Month"].str[:3]  # Extracts first 3 letters of the month

# Create dummy variables for months
month_dummies = pd.get_dummies(Walmart['months'])
Walmart1 = pd.concat([Walmart, month_dummies], axis=1)

# Visualization - Time plot
Walmart1.Footfalls.plot()

# Data Partition
Train = Walmart1.head(147)
Test = Walmart1.tail(12)

# Linear Model
linear_model = smf.ols('Footfalls ~ t', data=Train).fit()
pred_linear = pd.Series(linear_model.predict(Test['t']))
rmse_linear = np.sqrt(np.mean((Test['Footfalls'] - pred_linear) ** 2))

# Exponential Model
Exp = smf.ols('log_footfalls ~ t', data=Train).fit()
pred_Exp = pd.Series(Exp.predict(Test['t']))
rmse_Exp = np.sqrt(np.mean((Test['Footfalls'] - np.exp(pred_Exp)) ** 2))

# Quadratic Model
Quad = smf.ols('Footfalls ~ t + t_square', data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[['t', 't_square']]))
rmse_Quad = np.sqrt(np.mean((Test['Footfalls'] - pred_Quad) ** 2))

# Additive Seasonality
add_sea = smf.ols('Footfalls ~ Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec', data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']]))
rmse_add_sea = np.sqrt(np.mean((Test['Footfalls'] - pred_add_sea) ** 2))

# Multiplicative Seasonality
Mul_sea = smf.ols('log_footfalls ~ Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec', data=Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test[['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']]))
rmse_Mult_sea = np.sqrt(np.mean((Test['Footfalls'] - np.exp(pred_Mult_sea)) ** 2))

# Additive Seasonality Quadratic Trend
add_sea_Quad = smf.ols('Footfalls ~ t + t_square + Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov + Dec', data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 't', 't_square']]))
rmse_add_sea_quad = np.sqrt(np.mean((Test['Footfalls'] - pred_add_sea_quad) ** 2))

# Multiplicative Seasonality Linear Trend
Mul_Add_sea = smf.ols('log_footfalls ~ t + Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov + Dec', data=Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test[['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 't']]))
rmse_Mult_add_sea = np.sqrt(np.mean((Test['Footfalls'] - np.exp(pred_Mult_add_sea)) ** 2))

# RMSE Summary
data = {
    "MODEL": ["rmse_linear", "rmse_Exp", "rmse_Quad", "rmse_add_sea", "rmse_add_sea_quad", "rmse_Mult_sea", "rmse_Mult_add_sea"],
    "RMSE_Values": [rmse_linear, rmse_Exp, rmse_Quad, rmse_add_sea, rmse_add_sea_quad, rmse_Mult_sea, rmse_Mult_add_sea]
}
table_rmse = pd.DataFrame(data)
print(table_rmse)

# Predicting new values
predict_data = pd.read_excel(r"C:\Users\DELL\Desktop\Github\ML-Training\Predict_new.csv")
model_full = smf.ols('Footfalls ~ t + t_square + Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov + Dec', data=Walmart1).fit()
pred_new = pd.Series(model_full.predict(predict_data))
predict_data["forecasted_Footfalls"] = pred_new

# Autoregression Model (AR)
full_res = Walmart1.Footfalls - model_full.predict(Walmart1)

# ACF and PACF plots
tsa_plots.plot_acf(full_res, lags=12)
tsa_plots.plot_pacf(full_res, lags=12)

# AR model
model_ar = AutoReg(full_res, lags=1).fit()
pred_res = model_ar.predict(start=len(full_res), end=len(full_res) + len(predict_data) - 1, dynamic=False)
pred_res.reset_index(drop=True, inplace=True)

# Final Predictions using ASQT and AR(1) Model
final_pred = pred_new + pred_res
print(final_pred)
