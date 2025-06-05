from typing import List

import uvicorn
import numpy as np
import pandas as pd
from datetime import timedelta
from fastapi.responses import HTMLResponse
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from fastapi import FastAPI, HTTPException, Query, Body


app = FastAPI(title="Crude Oil Price Forecasting API",
              description="API for ARIMA and SARIMA time series forecasting")

def load_timeseries(file_path: str):
    return np.load(file_path)

@app.get("/", response_class=HTMLResponse)
def read_root():
    """API root with basic documentation"""
    return """
    <html>
        <head>
            <title>Crude Oil Price Forecasting API</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    line-height: 1.6;
                }
                h1, h2 {
                    color: #333;
                }
                code {
                    background-color: #f4f4f4;
                    padding: 2px 5px;
                    border-radius: 4px;
                }
                pre {
                    background-color: #f4f4f4;
                    padding: 10px;
                    border-radius: 4px;
                    overflow-x: auto;
                }
            </style>
        </head>
        <body>
            <h1>Crude Oil Price Forecasting API</h1>
            <p>This API provides ARIMA and SARIMA forecasting capabilities.</p>
            
            <h2>Endpoints:</h2>
            <ul>
                <li><code>GET /forecast?method=arima&steps=10</code> - Get ARIMA forecast</li>
                <li><code>GET /forecast?method=sarima&steps=10</code> - Get SARIMA forecast</li>
            </ul>
            
            <h2>Parameters:</h2>
            <ul>
                <li><code>method</code>: Either "arima" or "sarima"</li>
                <li><code>steps</code>: Number of steps to forecast (default: 10)</li>
                <li><code>order_p</code>: AR order (default: 1)</li>
                <li><code>order_d</code>: Differencing order (default: 1)</li>
                <li><code>order_q</code>: MA order (default: 1)</li>
                <li><code>seasonal_p</code>: Seasonal AR order (SARIMA only, default: 1)</li>
                <li><code>seasonal_d</code>: Seasonal differencing (SARIMA only, default: 1)</li>
                <li><code>seasonal_q</code>: Seasonal MA order (SARIMA only, default: 1)</li>
                <li><code>seasonal_m</code>: Seasonal period (SARIMA only, default: 12 for monthly basis)</li>
            </ul>
        </body>
    </html>
    """

@app.post("/forecast", response_class=HTMLResponse)
def forecast_time_series(
    method: str = Query(..., description="Forecasting method: 'arima' or 'sarima'"),
    steps: int = Query(10, description="Number of steps to forecast"),
    order_p: int = Query(4, description="AR order"),
    order_d: int = Query(0, description="Differencing order"),
    order_q: int = Query(4, description="MA order"),
    seasonal_p: int = Query(0, description="Seasonal AR order (SARIMA only)"),
    seasonal_d: int = Query(0, description="Seasonal differencing (SARIMA only)"),
    seasonal_q: int = Query(0, description="Seasonal MA order (SARIMA only)"),
    seasonal_m: int = Query(12, description="Seasonal period (SARIMA only)"),
    usd_rates: List[int] = Body(..., description='USD Exchange Rates')
):
    """
    Generate time series forecast using specified method
    """
    if method.lower() not in ["arima", "sarima"]:
        raise HTTPException(status_code=400, detail="Method must be either 'arima' or 'sarima'")
    
    ts_end = load_timeseries('data/preprocessed/end_diff.npy')
    ts_end_seasonal_diff = load_timeseries('data/preprocessed/end_seasonal_diff.npy')
    ts_exg = load_timeseries('data/preprocessed/exg_diff.npy')
    ts_exg_seasonal_diff = load_timeseries('data/preprocessed/exg_seasonal_diff.npy')
    
    try:
        if method.lower() == "arima":
            model = ARIMA(ts_end_seasonal_diff, ts_exg_seasonal_diff, order=(order_p, order_d, order_q))
            fitted_model = model.fit()
            
            forecast_result = fitted_model.get_forecast(steps=steps, exog=usd_rates)
            forecast_mean = forecast_result.predicted_mean
            forecast_conf_int = forecast_result.conf_int()
            
        else:  
            model = SARIMAX(
                ts_end,
                ts_exg, 
                order=(order_p, order_d, order_q),
                seasonal_order=(seasonal_p, seasonal_d, seasonal_q, seasonal_m)
            )
            fitted_model = model.fit(disp=False)
            
            forecast_result = fitted_model.get_forecast(steps=steps, exog=usd_rates)
            forecast_mean = forecast_result.predicted_mean
            forecast_conf_int = forecast_result.conf_int()
        
        last_date = pd.to_datetime('2024-01-01')
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=steps, freq='D')
        
        results = pd.DataFrame({
            'date': forecast_dates,
            'forecast': forecast_mean,
            'lower_ci': forecast_conf_int[:, 0],
            'upper_ci': forecast_conf_int[:, 1]
        })
        
        html_table = results.to_html(index=False, border=1, classes="table table-striped")
        
        html_content = f"""
        <html>
            <head>
                <title>{method.upper()} Forecast Results</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        margin: 40px;
                        line-height: 1.6;
                    }}
                    h1 {{
                        color: #333;
                    }}
                    .table {{
                        border-collapse: collapse;
                        width: 100%;
                        margin-top: 20px;
                    }}
                    .table-striped tr:nth-child(even) {{
                        background-color: #f2f2f2;
                    }}
                    th, td {{
                        padding: 8px;
                        text-align: left;
                        border: 1px solid #ddd;
                    }}
                    th {{
                        background-color: #4CAF50;
                        color: white;
                    }}
                </style>
            </head>
            <body>
                <h1>{method.upper()} Forecast Results</h1>
                <p>Forecast for the next {steps} days:</p>
                {html_table}
                <p><em>Model parameters: </em></p>
                <ul>
                    <li>ARIMA Parameters: ({order_p}, {order_d}, {order_q})</li>
                    {"<li>Seasonal Parameters: (" + str(seasonal_p) + ", " + str(seasonal_d) + ", " + str(seasonal_q) + ", " + str(seasonal_m) + ")</li>" if method.lower() == "sarima" else ""}
                </ul>
            </body>
        </html>
        """
        return HTMLResponse(content=html_content, status_code=200)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in forecast generation: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
