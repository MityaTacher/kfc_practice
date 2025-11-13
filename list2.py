from typing import Optional

import numpy as np
import pandas
import matplotlib.pyplot as plt

import hurst


def check_stationar(data_,
                    blocks: Optional[int] = None,
                    rows: Optional[int] = None
                    ):
    if blocks is not None and rows is not None:
        raise Exception('only one arg, because rows = length / blocks')
    if rows is blocks is None:
        raise Exception('must be one arg: "rows" or "blocks"')

    data_ = data_
    length = len(data_)

    if rows:
        blocks = int(length / rows)

    data_ = np.array_split(data_, blocks)

    mean = np.array([np.mean(part) for part in data_])
    disp = np.array([np.var(part, ddof=1) for part in data_])
    # disp = np.var(mean, ddof=1)
    avg_mean = np.average(mean)
    std_mean = np.std(mean, ddof=1)
    avg_disp = np.average(disp)
    std_disp = np.std(disp, ddof=1)

    return avg_mean, std_mean, avg_disp, std_disp
    # return avg_mean, std_mean, disp, std_disp


def nay(data_,
        blocks: Optional[int] = None,
        rows: Optional[int] = None
        ):
    if blocks is not None and rows is not None:
        raise Exception('only one arg, because rows = length / blocks')
    if rows is blocks is None:
        raise Exception('must be one arg: "rows" or "blocks"')

    data_ = data_
    length = len(data_)

    if rows:
        blocks = int(length / rows)

    data_ = np.array_split(data_, blocks)

    mean = np.array([np.mean(part) for part in data_])
    disp = np.var(mean, ddof=1)

    return disp


# тут короче начинается пизда
dataframe = pandas.read_csv("data/data.csv").iloc[:, 1].to_list()
TIME_ARRAY = np.array(dataframe)
TIME_ARRAY = TIME_ARRAY[TIME_ARRAY != 0]
TOTAL_LENGHT = len(TIME_ARRAY)
del dataframe

avg_mo, std_mo, avg_var, std_var = check_stationar(TIME_ARRAY, blocks=20)
deviation_mo = std_mo / avg_mo * 100
deviation_var = std_var / avg_var * 100

if deviation_mo > 10 or deviation_var > 40:
    TIME_ARRAY = np.diff(np.log10(TIME_ARRAY))  # разница меж соседними логарифмами

# --- начальные данные ---
# TIME_ARRAY = np.diff(np.log10(TIME_ARRAY))  # разница между соседними логарифмами
TIME_ARRAY = TIME_ARRAY[np.isfinite(TIME_ARRAY)]
TIME_ARRAY = TIME_ARRAY[TIME_ARRAY != 0]

# --- новые интервалы ---
m_values = [2 ** i for i in range(2, 18)]  # длины блоков
RS_values = []  # сюда будем писать средние R/S для каждого m


# --- функция для расчёта R/S для блока длины m ---
def nay(data_: np.ndarray, rows: int) -> np.floating:
    """
    Разбивает временной ряд на непересекающиеся блоки длины rows
    и считает среднее значение R/S по всем блокам.
    """
    n = len(data_)
    if rows >= n:
        raise ValueError("rows слишком велико, меньше длины ряда нужно")

    num_blocks = n // rows
    blocks = np.array_split(data_[:num_blocks * rows], num_blocks)
    RS_list = []

    for block in blocks:
        mean = np.mean(block)
        deviations = block - mean
        cum_dev = np.cumsum(deviations)
        R = np.max(cum_dev) - np.min(cum_dev)
        S = np.std(block, ddof=1)
        if S != 0:
            RS_list.append(R / S)

    return np.mean(RS_list)


# --- основная петля по m ---
for m in m_values:
    RS = nay(TIME_ARRAY, rows=m)
    RS_values.append(RS)

RS_values = np.array(RS_values)

# --- логарифмирование и аппроксимация ---
log_m = np.log10(m_values)
log_RS = np.log10(RS_values)

k, b = np.polyfit(log_m, log_RS, 1)  # линейная регрессия
H_my = k
c_my = np.exp(b)

# --- вывод ---
print(f"MY: H={H_my:.4f}, c={c_my:.4f}")

# --- визуализация ---
trend_line = np.polyval([k, b], log_m)
plt.figure()
plt.scatter(log_m, log_RS, s=50, label='log(R/S) точки')
plt.plot(log_m, trend_line, color='red', label=f'линия: y={k:.3f}x+{b:.3f}')
plt.xlabel('log(m)')
plt.ylabel('log(R/S)')
plt.grid()
plt.legend()
plt.show()

# --- сравнение с библиотекой ---
H_lib, c_lib, data = hurst.compute_Hc(TIME_ARRAY, kind='change', simplified=True)
print(f"hurst: H={H_lib:.4f}, c={c_lib:.4f}")

# --- сравнение графиков ---
plt.figure()
plt.loglog(data[0], data[1], 'o-', label='hurst.compute_Hc')
plt.loglog(m_values, RS_values, 's-', label='my R/S')
plt.loglog(m_values, c_my * np.array(m_values) ** H_my, '--', label='my fit')
plt.loglog(data[0], c_lib * np.array(data[0]) ** H_lib, ':', label='hurst fit')
plt.legend()
plt.grid()
plt.show()
