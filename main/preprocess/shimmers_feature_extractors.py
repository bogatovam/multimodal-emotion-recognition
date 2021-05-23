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
        df = df[['Analog AccelX_raw', 'Analog AccelY_raw', 'Analog AccelZ_raw', 'GSR Range_raw', 'GSR_raw', 'GyroX_raw',
                 'GyroY_raw', 'GyroZ_raw', 'PPT_raw', 'Timestamp_CAL[sec]', 'PPT_CAL[mV]', 'GSR_CAL[kOhm]',
                 'Analog AccelX_CAL[m/s^2]',
                 'Analog AccelY_CAL[m/s^2]', 'Analog AccelZ_CAL[m/s^2]', 'GyroX_CAL[1/s]', 'GyroY_CAL[1/s]',
                 'GyroZ_CAL[1/s]']]
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

        df = df.dropna()

        max_ = df['Timestamp_CAL[sec]'].max()
        grouped = df.groupby(pd.cut(df['Timestamp_CAL[sec]'], np.arange(0, max_ + step, step))).mean().drop(
            'Timestamp_CAL[sec]', axis=1)
        np_array = np.nan_to_num(grouped.to_numpy())

        # stand & norm
        np_array = np.array(list(map(stand, np_array)))
        min_max_scaler = preprocessing.MinMaxScaler()
        return min_max_scaler.fit_transform(np_array)
    except FileNotFoundError:
        return np.asarray([])


def stand(x):
    return (x - np.mean(x) / np.std(x)) if np.std(x) != 0.0 else x
