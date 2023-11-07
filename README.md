# LSTM_Forecasting

## Problem Statement - 

The current dataset contains the number of observed scanned receipts each day of 2021. The goal is to develop an algorithm which can predict the approximate number of scanned receipts for each month of 2022. Also, run an inference procedure against the trained model and return the predicted results.

## My Approach - 

This project hosts a Python Flask web application that forecasts daily receipt counts. It uses an LSTM model to predict future values based on historical data. 

1. Stationarity Transformation

LSTMs require stationary data to function optimally. We address the non-stationary nature of our dataset through first-order differentiation, which helps in normalizing trends and seasonality, making the data suitable for our model.

2. Data Normalization

To enhance LSTM performance, we scale the dataset to a range of [-1,1]. This step is crucial for efficient learning and stability during the network's weight update process.

3. Lag Selection

The 'lag' parameter determines the memory of our LSTM model. Utilizing the PACF plot, we identify the optimal lag that informs the model how far back it should look to forecast future receipts effectively.

4. Time Series Data Preparation

A time series generator organizes our data into sequences based on the chosen lag value. This structure is essential for training the LSTM, ensuring it receives data in a format that reflects temporal dynamics.

5. Model Training and Serialization

Once compiled with an appropriate loss function and adam optimizer, the LSTM is trained on the processed data. The trained model is then saved and predictions are made.

### About the application - 

1. The main page displays the actual vs. forecasted receipt counts for Oct 2021 - Dec 2021 and provides a Mean Absolute Percentage Error (MAPE) for the forecasted values. This page also provides an interactive visualization (using plotly).

2. The button '/forecast_2022' provides a monthly receipt count forecast for the year 2022 with a visualization of the predicted receipt counts.
   
## How to run - 

1. If you have docker installed, this code can be pulled down via Docker Hub by executing the following commands -
    1. docker pull karans98/forecast-app:latest
    2. docker run -p 3000:3000 karans98/forecast-app:latest

    (The website can be accessed at localhost:3000)

2. If you do not have docker installed then you can git clone the repository, install all the dependencies and run the python app
    1. Clone this repository :[git clone https://github.com/karans98/LSTM_Forecasting.git]
    2. Once cloning is complete, navigate to the cloned repository :[cd LSTM_Forecasting]
    3. Install the Python libraries mentioned in the requirements.txt file :[pip install -r requirements.txt]
    4. Run the Python file :[python app.py]
    5. Click the link and view the results

### Important Notes - 

- The initial page load may take a few seconds due to the rendering of interactive plots.
- The "Forecast 2022" feature initiates a computation-intensive process to forecast the next 365 days. Please allow additional time for this process to complete and for the results to be displayed.

