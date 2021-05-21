import pandas as pd
import numpy as np
from sklearn import preprocessing

sensors = ['Analog AccelX_raw', 'Analog AccelY_raw', 'Analog AccelZ_raw',
           'GSR Range_raw', 'GSR_raw', 'GyroX_raw', 'GyroY_raw', 'GyroZ_raw',
           'PPT_raw', 'PPT_CAL[mV]', 'GSR_CAL[kOhm]', 'Analog AccelX_CAL[m/s^2]',
           'Analog AccelY_CAL[m/s^2]', 'Analog AccelZ_CAL[m/s^2]',
           'GyroX_CAL[1/s]', 'GyroY_CAL[1/s]', 'GyroZ_CAL[1/s]']


def open_shimmers(filename: str, elements_per_sec=10) -> np.ndarray:
    try:
        df = pd.read_csv(filename, index_col=None, header=0)
        df = df.sort_values('Timestamp_CAL[sec]')
        step = 1 / elements_per_sec
        max_ = df['Timestamp_CAL[sec]'].max()
        grouped = df.groupby(pd.cut(df['Timestamp_CAL[sec]'], np.arange(0, max_ + step, step))).mean().drop(
            'Timestamp_CAL[sec]', axis=1)

        # stand & norm
        grouped = (grouped - grouped.mean(axis=0)) / grouped.std(axis=0)
        min_max_scaler = preprocessing.MinMaxScaler()
        grouped = pd.DataFrame(min_max_scaler.fit_transform(grouped))
        return grouped.to_numpy()
    except FileNotFoundError:
        return np.asarray([])
