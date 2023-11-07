import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, render_template
import plotly.graph_objs as go
from plotly.utils import PlotlyJSONEncoder
import json

#Data
data = pd.read_csv('https://fetch-hiring.s3.amazonaws.com/machine-learning-engineer/receipt-count-prediction/data_daily.csv', parse_dates = ['# Date'], index_col = '# Date')

#Differentiate data to remove stationarity
data_diff = data.diff().dropna()

#Train-Test split
train = data_diff.iloc[:-92] # First 9 months
test = data_diff.iloc[-92:] # Last 3 months

#Scale the values between [-1,1]
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(train.values.reshape(-1,1))
scaled_train = scaler.transform(train.values.reshape(-1,1))
scaled_test = scaler.transform(test.values.reshape(-1,1))

#Load forecasting model
pred_model = load_model('forecast_model')

look_back = 10 #Based on PACF plot, look at 10 previous values to predict the future
n_features = 1 #Since it is a univariate series

#Function to forecast values
def forecasted_values(no_of_days, train_data, last_known_value, n_input, n_features = 1):
  test_predictions = []
  first_eval_batch = train_data[-n_input:]
  current_batch = first_eval_batch.reshape((1, n_input, n_features))

  for i in range(no_of_days):
      current_pred = pred_model.predict(current_batch)[0]
      test_predictions.append(current_pred)
      current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1) #Update batch by removing the first value and adding the last one

  test_predict_scaled = scaler.inverse_transform(test_predictions)   # Invert predictions
  
  # To reverse the differentiation that was perfomed
  forecasted_values = [last_known_value] 
  for predicted_difference in test_predict_scaled:
      next_value = forecasted_values[-1] + predicted_difference
      forecasted_values.append(next_value)

  return forecasted_values

def get_data_df(forecast_list, forecast_index):
  forecast_data = [i[0] for i in forecast_list[1:]]
  forecast_df = pd.DataFrame({'Date': forecast_index, 'Receipt_Count': forecast_data})
  forecast_df.set_index('Date', inplace=True)
  return forecast_df

app = Flask(__name__)

@app.route('/')
def home():
    last_known_value = data.iloc[-93]['Receipt_Count'] #Last value of the training data
    test_predicts = forecasted_values(len(test), scaled_train, last_known_value, look_back)
    test_index_2021 = pd.date_range(start='2021-10-01', end='2021-12-31', freq='D')
    test_df = get_data_df(test_predicts,test_index_2021)
    actual_test = data.iloc[-92:]

    mape = round(np.mean(np.abs((np.array(actual_test['Receipt_Count']) - np.array(test_df['Receipt_Count'])) / np.array(actual_test['Receipt_Count']))) * 100,2)

    # The plotly plot
    trace1 = go.Scatter(x=actual_test.index, y=actual_test['Receipt_Count'], mode='lines', name='Actual data')
    trace2 = go.Scatter(x=test_df.index, y=test_df['Receipt_Count'], mode='lines', name='Forecasted data')

    fig = go.Figure(data=[trace1, trace2], layout=go.Layout(title='Actual vs Forecasted Receipt counts for Oct 2021 - Dec 2021',
                                                            xaxis=dict(title='Date'), yaxis=dict(title='Receipt Count'), 
                                                            width= 1000, height=500, autosize=False))
    graphJSON = json.dumps(fig, cls=PlotlyJSONEncoder)

    show_heading = True
    
    return render_template('index.html', graphJSON1 = graphJSON, output = mape, show_heading = show_heading)

@app.route('/forecast_2022', methods=['POST'])
def forecast_2022():
    last_known_value = data.iloc[-1]['Receipt_Count'] #Last value of the training data
    predicts_2022 = forecasted_values(365, scaled_test, last_known_value, look_back)
    date_index_2022 = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
    forecast_df_2022 = get_data_df(predicts_2022,date_index_2022)
    monthly_forecast_2022 = forecast_df_2022.resample('M').sum()
    monthly_forecast_2022.index = monthly_forecast_2022.index.strftime('%B')
    monthly_forecast_2022['Receipt_Count'] = (monthly_forecast_2022['Receipt_Count']/1000000).round().astype(int)
    monthly_forecast_2022.rename(columns={'Receipt_Count': 'Receipt Count in Millions'}, inplace=True)

    df_html = monthly_forecast_2022.to_html()

    # The plotly plot
    trace1 = go.Scatter(x=data.index, y=data['Receipt_Count'], mode='lines', name='Actual data')
    trace2 = go.Scatter(x=forecast_df_2022.index, y=forecast_df_2022['Receipt_Count'], mode='lines', name='Forecasted data')

    fig = go.Figure(data=[trace1, trace2], layout=go.Layout(title='Actual & Forecasted Receipt counts for 2021 & 2022',
                                                            xaxis=dict(title='Date'), yaxis=dict(title='Receipt Count')))
    graphJSON = json.dumps(fig, cls=PlotlyJSONEncoder)

    show_line = True
    return render_template('index.html', table = df_html,graphJSON2 = graphJSON, show_line=show_line)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port = int("3000"))


