import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from prophet.diagnostics import cross_validation, performance_metrics
import pickle

### Load and Prepare Data for Prophet ###
df = pd.read_csv("../data/cleaned_data/cleaned_amazon_sales_dataset.csv")
df['order_date'] = pd.to_datetime(df['order_date'])
prophet_df = df.groupby('order_date')['total_revenue'].sum().reset_index()
prophet_df.columns = ['ds', 'y']  

print(prophet_df.head())



### Fitting the Model ###
model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
model.fit(prophet_df)
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())



### Visualize the forecast ###
fig1 = model.plot(forecast)
plt.title("Amazon Toplam Gelir Tahmini")
plt.xlabel("Tarih")
plt.ylabel("Toplam Gelir (Revenue)")
plt.show()

fig2 = model.plot_components(forecast)
plt.show()



### Model Evaluation with Cross-Validation ###
df_cv = cross_validation(model, initial='540 days', period='31 days', horizon='90 days')

df_p = performance_metrics(df_cv)

print(df_p[['horizon', 'mae', 'rmse', 'mape']].head())



### Final test on the last 100 days of data ###
train_df = prophet_df[:-100] 
test_df = prophet_df[-100:]  

m_test = Prophet(yearly_seasonality=True, weekly_seasonality=True)
m_test.fit(train_df)

future_test = m_test.make_future_dataframe(periods=100)
forecast_test = m_test.predict(future_test)

plt.figure(figsize=(12, 6))
plt.plot(test_df['ds'], test_df['y'], label='Actual Data', color='black')
plt.plot(forecast_test['ds'].tail(100), forecast_test['yhat'].tail(100), label='Forecast', color='blue')
plt.fill_between(forecast_test['ds'].tail(100), forecast_test['yhat_lower'].tail(100), forecast_test['yhat_upper'].tail(100), color='blue', alpha=0.2)
plt.legend()
plt.title("Last 100 Days Forecast vs Actuals")
plt.show()



### Save the model for future use ###
with open('prophet_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model 'prophet_model.pkl' olarak kaydedildi!")