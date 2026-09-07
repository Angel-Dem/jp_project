import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
import warnings
import itertools

data = pd.read_csv('Nat_Gas.csv')
data["Dates"] = pd.to_datetime(data["Dates"])
data.sort_values(["Dates"])
data["Prices"] = pd.to_numeric(data["Prices"], errors='coerce')
data = data.dropna() 

data["Dates"] = pd.to_datetime(data["Dates"])
df = data.set_index("Dates").copy()
df = df.dropna()
ts_diff = df.diff().dropna()
data = data.set_index('Dates')['Prices'] 
T = len(ts_diff.dropna())
max_lags = int(min(10 * np.log10(T), T // 4))


conf = 1.96 / np.sqrt(T)

acf_vals = acf(ts_diff.dropna(), nlags=max_lags)
pacf_vals = pacf(ts_diff.dropna(), nlags=max_lags)

# лаги, значимо отличные от нуля -> кандидаты в q и p соответственно
q_candidates = [i for i in range(1, len(acf_vals)) if abs(acf_vals[i]) > conf]
p_candidates = [i for i in range(1, len(pacf_vals)) if abs(pacf_vals[i]) > conf]

p_range = range(0, max(p_candidates, default=1) + 1)
q_range = range(0, max(q_candidates, default=1) + 1)

def aicc(results, p, q):
    """AICc из материала — поправка для коротких рядов."""
    T = results.nobs
    k = p + q + 1
    ll = results.llf
    aic = -2 * ll + 2 * k
    if T - k - 2 <= 0:
        return np.inf  # поправка не определена при слишком коротком ряде
    return aic + (2 * k * (k + 1)) / (T - k - 2)

best_crit = np.inf
best_order = None
best_model = None
d=1
for p, q in itertools.product(p_range, q_range):
    order = (p, d, q)
    try:
        results = ARIMA(data, order=order).fit()
        crit = aicc(results, p, q)  # можно заменить на results.aic или results.bic
        if crit < best_crit:
            best_crit = crit
            best_order = order
            best_model = results
    except Exception as e:
        print(order, "FAILED:", repr(e))

final_model = ARIMA(data, order=best_order)
results = final_model.fit()

forecast_values = results.forecast(steps=18)

n_periods = 18  # столько точек вперёд вы прогнозировали на графике

forecast_res = results.get_forecast(steps=n_periods)

forecast_df = pd.DataFrame({
    'forecast': forecast_res.predicted_mean
})
forecast_df.to_csv('forecast.csv')
