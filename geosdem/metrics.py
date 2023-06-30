

def mape(actual: list[float], forecast: list[float]) -> float:
    """_summary_

    Args:
        actual (List[float]): _description_
        forecast (List[float]): _description_

    Returns:
        float: _description_
    # Example usage
    actual = [100, 125, 150, 175]
    forecast = [110, 130, 145, 160]
    print(mape(actual, forecast))  # Output: 0.15
    credit to : https://medium.com/@vinaychaudhari1996/the-top-11-timeseries-forecasting-metrics-you-need-to-know-mape-smape-me-mae-rmse-mse-5023fd26aa88
    """    
    n = len(actual)
    mape = sum(abs(a - f) / a for a, f in zip(actual, forecast)) / n
    return mape

def me(actual: list[float], forecast: list[float]) -> float:
    """_summary_

    Args:
        actual (List[float]): _description_
        forecast (List[float]): _description_

    Returns:
        float: _description_
    # Example usage
    actual = [100, 125, 150, 175]
    forecast = [110, 130, 145, 160]
    print(me(actual, forecast))  # Output: -7.5
    """    
    n = len(actual)
    me = sum(a - f for a, f in zip(actual, forecast)) / n
    return me

def mae(actual: list[float], forecast: list[float]) -> float:
    """_summary_

    Args:
        actual (list[float]): _description_
        forecast (list[float]): _description_

    Returns:
        float: _description_
    # Example usage
    actual = [100, 125, 150, 175]
    forecast = [110, 130, 145, 160]
    print(mae(actual, forecast))  # Output: 15.0
    """    
    n = len(actual)
    mae = sum(abs(a - f) for a, f in zip(actual, forecast)) / n
    return mae

def mae(actual: list[float], forecast: list[float]) -> float:
    """_summary_

    Args:
        actual (list[float]): _description_
        forecast (list[float]): _description_

    Returns:
        float: _description_
    # Example usage
    actual = [100, 125, 150, 175]
    forecast = [110, 130, 145, 160]
    print(mae(actual, forecast))  # Output: 15.0
    """    
    n = len(actual)
    mae = sum(abs(a - f) for a, f in zip(actual, forecast)) / n
    return mae

import math

def rmse(actual: list[float], forecast: list[float]) -> float:
    """_summary_

    Args:
        actual (list[float]): _description_
        forecast (list[float]): _description_

    Returns:
        float: _description_
    # Example usage
    actual = [100, 125, 150, 175]
    forecast = [110, 130, 145, 160]
    print(rmse(actual, forecast))  # Output: 17.6776
    """    
    n = len(actual)
    rmse = math.sqrt(sum((a - f)**2 for a, f in zip(actual, forecast)) / n)
    return rmse

def mse(actual: list[float], forecast: list[float]) -> float:
    """_summary_

    Args:
        actual (list[float]): _description_
        forecast (list[float]): _description_

    Returns:
        float: _description_
    # Example usage
    actual = [100, 125, 150, 175]
    forecast = [110, 130, 145, 160]
    print(mse(actual, forecast))  # Output: 308.3333
    """    
    n = len(actual)
    mse = sum((a - f)**2 for a, f in zip(actual, forecast)) / n
    return mse

def msle(actual: list[float], forecast: list[float]) -> float:
    """_summary_

    Args:
        actual (list[float]): _description_
        forecast (list[float]): _description_

    Returns:
        float: _description_
    # Example usage
    actual = [100, 125, 150, 175]
    forecast = [110, 130, 145, 160]
    print(msle(actual, forecast))  # Output: 308.3333
    """    
    n = len(actual)
    msle = sum((math.log(1 + a) - math.log(1 + f))**2 for a, f in zip(actual, forecast)) / n
    return msle

def mase(actual: list[float], forecast: list[float], seasonality: int) -> float:
    """_summary_

    Args:
        actual (list[float]): _description_
        forecast (list[float]): _description_
        seasonality (int): _description_

    Returns:
        float: _description_
    # Example usage
    actual = [100, 125, 150, 175, 200, 225]
    forecast = [110, 130, 145, 160, 180, 190]
    seasonality = 3  # In this example, the seasonality is 3 time periods
    print(mase(actual, forecast, seasonality))  # Output: 0.5
    """    
    n = len(actual)
    abs_differences = sum(abs(a - f) for a, f in zip(actual, forecast))
    abs_differences_in_seasonality = sum(abs(a - b) for a, b in zip(actual[seasonality:], actual[:-seasonality]))
    mase = abs_differences / abs_differences_in_seasonality
    return mase

from statistics import median

def medae(actual: list[float], forecast: list[float]) -> float:
    """_summary_

    Args:
        actual (list[float]): _description_
        forecast (list[float]): _description_

    Returns:
        float: _description_
    # Example usage
    actual = [100, 125, 150, 175]
    forecast = [110, 130, 145, 160]
    print(medae(actual, forecast))  # Output: 15.0
    """    
    abs_errors = [abs(a - f) for a, f in zip(actual, forecast)]
    medae = median(abs_errors)
    return medae

from statistics import median

def mdrae(actual: list[float], forecast: list[float]) -> float:
    """_summary_

    Args:
        actual (list[float]): _description_
        forecast (list[float]): _description_

    Returns:
        float: _description_
    # Example usage
    actual = [100, 125, 150, 175]
    forecast = [110, 130, 145, 160]
    print(mdrae(actual, forecast))  # Output: 0.133333
    
    """    
    rel_abs_errors = [abs(a - f) / a for a, f in zip(actual, forecast)]
    mdrae = median(rel_abs_errors)
    return mdrae



import math

def gmrae(actual: list[float], forecast: list[float]) -> float:
    """_summary_

    Args:
        actual (list[float]): _description_
        forecast (list[float]): _description_

    Returns:
        float: _description_
    # Example usage
    actual = [100, 125, 150, 175]
    forecast = [110, 130, 145, 160]
    print(gmrae(actual, forecast))  # Output: 0.131371
    
    """    
    rel_abs_errors = [abs(a - f) / a for a, f in zip(actual, forecast)]
    gmrae = math.prod(rel_abs_errors)**(1 / len(rel_abs_errors))
    return gmrae

def smape(actual: list[float], forecast: list[float]) -> float:
    """SMAPE is similar to MAPE, but it is calculated as the average of the absolute

    Args:
        actual (list[float]): _description_
        forecast (list[float]): _description_

    Returns:
        float: _description_
    # Example usage
    actual = [100, 125, 150, 175]
    forecast = [110, 130, 145, 160]
    print(smape(actual, forecast))  # Output: 0.15
    
    """    
    n = len(actual)
    smape = sum(abs(a - f) / (abs(a) + abs(f)) for a, f in zip(actual, forecast)) / (2 * n)
    return smape


