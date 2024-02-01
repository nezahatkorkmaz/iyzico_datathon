import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm

df = pd.read_csv("/kaggle/input/inflation/train_with_inflation.csv")
res = pd.read_csv("/kaggle/input/iyzico-datathon/sample_submission.csv")
res[['month_id', 'merchant_id']] = res['id'].str.extract(r'(\d{6})(merchant_\d+)')

# Sort the DataFrame by 'merchant_id' and 'month_id' in descending order
df_sorted = df.sort_values(by=['merchant_id', 'month_id_x'], ascending=[True, False])

# Group by 'merchant_id' and keep the first row of each group
latest_data = df_sorted.groupby('merchant_id').first().reset_index()

result = pd.merge(res, latest_data, on='merchant_id', how='left')

# Ignoring small values and churned customers
result.loc[(result['net_payment_count_y'] < 5) | (result['month_id_y'] < 202309), 'net_payment_count_x'] = 0

# Get the latest data for baseline
result.loc[~((result['net_payment_count_y'] < 5) | (result['month_id_y'] < 202309)), 'net_payment_count_x'] = result['net_payment_count_y']

ids = result[result['net_payment_count_x'] != 0]['merchant_id'].unique()
df['month_id_x'] = pd.to_datetime(df['month_id_x'], format='%Y%m')
df.set_index('month_id_x', inplace=True)

from numpy.linalg import LinAlgError

for id in tqdm(ids):
    # Grouping by 'merchant_id'
    df_x = df.groupby('merchant_id').get_group(id)

    # Check if there is sufficient data to fit the model
    if len(df_x) > 3:
        try:
            model = ARIMA(df_x['net_payment_count'], order=(1, 1, 1))
            model_fit = model.fit()
            forecast_result = model_fit.get_forecast(steps=3)
            forecast = forecast_result.predicted_mean
            filtered_res = result[result['merchant_id']==id].sort_values(by='month_id_x')['net_payment_count_x']
            result.loc[filtered_res.index[:3], 'net_payment_count_x'] = list(forecast)
        except LinAlgError:
            print(f"Linear algebra error for merchant_id {id}")
        except Exception as e:
            print(f"Error fitting model for merchant_id {id}: {e}")

columns_to_keep = ['id', 'net_payment_count_x']
result = result[columns_to_keep]
