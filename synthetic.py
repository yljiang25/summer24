import pandas as pd
import ruptures as rpt
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from matplotlib import pyplot

def change_point_detection(data: pd.Series, model: str, min_size: int, penalty: int):
    """
    This function uses the PELT (Pruned Exact Linear Time) function
    of the Ruptures library to analyze the given time-series data
    for change point detection.\n
    :param data: A Pandas Series containing the time-series data whose change points need to be detected.\n
    :type data: pd.Series\n
    :param model: The model used by PELT to perform the analysis. Allowed types include "l1", "l2", and "rbf".\n
    :type model: str\n
    :param min_size: The minimum separation (time steps) between two consecutive change points detected by the model.\n
    :type min_size: int\n
    :param penalty: The penalty value used during prediction of change points.\n
    :type penalty: int\n
    :return: Returns a sorted list of breakpoints.\n
    :rtype: list\n
    """
    data_np = data.to_numpy()
    algo = rpt.Pelt(model=model, min_size=min_size).fit(data_np)
    change_points = algo.predict(pen=penalty)
    # return result
    
    
    
    
    valid_change_points = [cp for cp in change_points if cp < len(data)]

    # Convert valid change points to the corresponding times
    change_points_times = data.index[valid_change_points]

    # Sort change points by time to connect them in order
    sorted_change_points_times = sorted(change_points_times)

    # Extract the corresponding temperature values for the sorted change points
    sorted_change_points_values = data.loc[sorted_change_points_times]

    plt.figure(figsize=(20, 6))
    plt.plot(data.index, data.values, linestyle = "dashed", label='Temperature')
    plt.vlines(x = sorted_change_points_times, ymin = 70, ymax = 100,
            colors = 'red',
            label = 'Change points')
    plt.title('Synthetic Temp (F) Change Point Detection')
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.legend()
    plt.show()
    
def detect_seasonality(data: pd.Series, model_type: str = "additive") -> pd.DataFrame:
    """
    This function analyzes a given time-series data for seasonality.\n
    :param data: A Pandas Series, assumed to have dates as its indices with the corresponding
        values of the time-series data.\n
    :type data: pd.Series\n
    :param model_type: Can be "additive" or "multiplicative", determines the type of seasonality
        model assumed for the data.\n
    :type model_type: str\n
    :return: Returns a Pandas DataFrame that contain the Trend, Seasonal, and Residual components
        computed using the given model type. Can be plotted using the "plot" method of Pandas DataFrame class.\n
    :rtype: pd.DataFrame\n
    """
    decompose_result = seasonal_decompose(data, model=model_type, period=60)
    decompose_result.plot()
    plt.show()
    # return decompose_result

df = pd.read_csv('Original_and_Synthetic_Temp_Data.csv')

# Assuming 'Time' is the index, convert it to a datetime object
df['Time'] = pd.to_datetime(df['Time'])
df.set_index('Time', inplace=True)

syn_temp = df['Synthetic Temp (F)']

# change_point_detection(syn_temp, model="l2", min_size=250, penalty=10)
change_point_detection(syn_temp, model="l2", min_size=250, penalty=20) ##star





# detect_seasonality(syn_temp, "Multiplicative")
# detect_seasonality(syn_temp, "Additive")
