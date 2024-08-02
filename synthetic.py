import pandas as pd
import ruptures as rpt
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from matplotlib import pyplot

def change_point_detection(data: pd.Series, model: str, min_size: int, penalty: int, model_type: str):
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
    
    valid_change_points = [cp for cp in change_points if cp < len(data)]
    # Convert valid change points to the corresponding times
    change_points_times = data.index[valid_change_points]

    plt.figure(figsize=(12, 9))
    plt.subplot(4, 1, 1)
    plt.plot(data.index, data.values, linestyle = "dashed", label='Temperature')
    plt.vlines(x = change_points_times, ymin = 70, ymax = 95,
            colors = 'red',
            label = 'Change points')
    plt.title('Synthetic Temp (F) Change Point Detection')
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.legend()
    
    decompose_result = seasonal_decompose(data, model=model_type, period=60)
    plt.subplot(4, 1, 2)
    plt.plot(decompose_result.trend, label='Trend', color='r')
    plt.ylabel('Trend')
    plt.title('Synthetic Temp (F) Seasonality analysis')
    plt.subplot(4, 1, 3)
    plt.plot(decompose_result.seasonal, label='Seasonal', color='g')
    plt.ylabel('Seasonal')
    plt.subplot(4, 1, 4)
    plt.plot(decompose_result.resid, label='Residual', color='b')
    plt.ylabel('Residual')
    plt.tight_layout()
    plt.show()
    return change_points_times
    
def detect_seasonality(data: pd.Series, model_type: str):
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
    plt.figure(figsize = (13, 6))
    plt.subplot(3, 1, 1)
    plt.plot(decompose_result.trend, label='Trend', color='r')
    plt.ylabel('Trend')
    plt.title('Synthetic Temp (F) Breakpoint Seasonality')
    plt.subplot(3, 1, 2)
    plt.plot(decompose_result.seasonal, label='Seasonal', color='g')
    plt.ylabel('Seasonal')
    plt.subplot(3, 1, 3)
    plt.plot(decompose_result.resid, label='Residual', color='b')
    plt.ylabel('Residual')
    plt.tight_layout()
    plt.show()
    
    # decompose_result1 = seasonal_decompose(data1, model=model_type, period=60)
    # plt.figure(figsize = (20, 6))
    # plt.subplot(3, 2, 1)
    # plt.plot(decompose_result1.trend, label='Trend', color='r')
    # plt.ylabel('Trend')
    # plt.title('Synthetic Temp (F) Breakpoint seasonality')
    # plt.subplot(3, 2, 3)
    # plt.plot(decompose_result1.seasonal, label='Seasonal', color='g')
    # plt.ylabel('Seasonal')
    # plt.subplot(3, 2, 5)
    # plt.plot(decompose_result1.resid, label='Residual', color='b')
    # plt.ylabel('Residual')
    # # plt.plot(decompose_result.trend, color='r')
    # plt.tight_layout()
    
    # decompose_result2 = seasonal_decompose(data2, model=model_type, period=60)
    # # plt.figure(figsize = (10, 6))
    # plt.subplot(3, 2, 2)
    # plt.plot(decompose_result2.trend, label='Trend', color='r')
    # plt.ylabel('Trend')
    # # plt.title('Synthetic Temp (F) Breakpoint seasonality')
    # plt.subplot(3, 2, 4)
    # plt.plot(decompose_result2.seasonal, label='Seasonal', color='g')
    # plt.ylabel('Seasonal')
    # plt.subplot(3, 2, 6)
    # plt.plot(decompose_result2.resid, label='Residual', color='b')
    # plt.ylabel('Residual')
    # # plt.plot(decompose_result.trend, color='r')
    # plt.tight_layout()
    
    # plt.show()
    
    
def trend_seasonality(syn_data: pd.Series, og_data: pd.Series, model_type: str):
    syn_result = seasonal_decompose(syn_data, model=model_type, period=60)
    og_result = seasonal_decompose(og_data, model=model_type, period=60)
    
    plt.figure(figsize = (10, 6))
    plt.plot(og_result.trend, label='Original Temp Trend', color='r')
    plt.plot(syn_result.trend, label='Synthetic Temp Trend', color='g')
    plt.title("Comparing Synthetic Temp Breakpoint Trends Against Original Temp")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
df = pd.read_csv('Original_and_Synthetic_Temp_Data.csv')

# Assuming 'Time' is the index, convert it to a datetime object
df['Time'] = pd.to_datetime(df['Time'])
df.set_index('Time', inplace=True)

syn_temp = df['Synthetic Temp (F)']
og_temp = df['Original Temp (F)']

# change_point_detection(syn_temp, model="l2", min_size=250, penalty=10)
# change_point_detection(syn_temp, model="l2", min_size=250, penalty=20) ##star

breaks = change_point_detection(syn_temp, model="l2", min_size=250, penalty=20, model_type="Multiplicative") ##star
for i in range(len(breaks)-1):
    segment = syn_temp[breaks[i]:breaks[i+1]]
    # segment2 = syn_temp[breaks[i+1]:breaks[i+2]]
    # detect_seasonality(segment, model_type="Multiplicative")
    
    og_seg = og_temp[breaks[i]:breaks[i+1]]
    trend_seasonality(segment, og_seg, "Multiplicative")

# detect_seasonality(syn_temp, "Multiplicative")
# detect_seasonality(syn_temp, "Additive")
