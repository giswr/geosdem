"""Main module."""
from .geosdem import *




import numpy as np
import xarray as xr
from sklearn.ensemble import RandomForestRegressor

def predict_rainfall_3d(rainfall_data, temp_data, humidity_data, prediction_data):
    """
    Predict 3D netCDF rainfall with independent variables like temperature and humidity netCDF.
    
    Args:
        rainfall_data (ndarray): xarray Dataset of historical rainfall data
        temp_data (str): xarray Dataset of historical temperature data
        humidity_data (ndarray): xarray Dataset of historical humidity data
        prediction_data (ndarray): xarray Dataset of input data to predict rainfall
    
    Returns:
        ndarray: A 3D xarray Dataset containing the predicted rainfall values for the input data
    """
    # Extract the input and output variables from the rainfall data
    X = np.stack([temp_data.values.flatten(), humidity_data.values.flatten()], axis=1)
    y = rainfall_data.values.flatten()
    
    # Train a Random Forest model on the historical data
    rf = RandomForestRegressor(n_estimators=100, random_state=0)
    rf.fit(X, y)
    
    # Make predictions on the input data
    temp_pred = prediction_data['temperature'].values.flatten()
    humidity_pred = prediction_data['humidity'].values.flatten()
    X_pred = np.stack([temp_pred, humidity_pred], axis=1)
    rainfall_pred = rf.predict(X_pred)
    rainfall_pred = np.reshape(rainfall_pred, prediction_data['rainfall'].shape)
    
    # Convert the predicted rainfall values to an xarray Dataset
    dims = prediction_data['rainfall'].dims
    coords = prediction_data['rainfall'].coords
    rainfall_pred = xr.DataArray(rainfall_pred, dims=dims, coords=coords)
    rainfall_pred = xr.Dataset({'rainfall': rainfall_pred})
    
    return rainfall_pred