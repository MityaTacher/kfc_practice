from typing import Optional

import numpy as np
import pandas
import matplotlib.pyplot as plt

import hurst


def nay1(data_,
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

avg_mo, std_mo, avg_var, std_var = nay1(TIME_ARRAY, blocks=20)
deviation_mo = std_mo / avg_mo * 100
deviation_var = std_var / avg_var * 100

# if deviation_mo > 10 or deviation_var > 40:
#     TIME_ARRAY = np.diff(np.log10(TIME_ARRAY))  # разница меж соседними логарифмами

TIME_ARRAY = np.diff(np.log10(TIME_ARRAY))  # разница меж соседними логарифмами
m = [2 ** i for i in range(2, 8)]
m_var = []

for i in m:
    avg = nay(TIME_ARRAY, rows=i)
    m_var.append(avg)

m_var = np.array(m_var)
m_log = np.log10(m)
m_var_log = np.log10(m_var)

coefs = np.polyfit(m_log, m_var_log, 1)  # линейная аппроксимация
k, b = coefs
trend_line = np.polyval(coefs, m_log)

ss_res = np.sum((m_var_log - trend_line) ** 2)
ss_tot = np.sum((m_var_log - np.mean(m_var_log)) ** 2)
r2 = 1 - ss_res / ss_tot

hurst1 = 1 + k / 2
print(f'H: {hurst1}')

plt.gcf().canvas.manager.set_window_title("Rostics2")
plt.scatter(m_log, m_var_log, s=50)
plt.plot(m_log, trend_line, color='red', label=f'y={k:.3f}x+{b:.3f}\nR2={r2:.3f}')

plt.legend()
plt.grid()
plt.show()

H, c, data = hurst.compute_Hc(TIME_ARRAY, kind='change', simplified=True)

# Plot
f, ax = plt.subplots()
ax.plot(data[0], c * data[0] ** H, color="deepskyblue")
ax.scatter(data[0], data[1], color="purple")
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Time interval')
ax.set_ylabel('R/S ratio')
ax.grid(True)
plt.show()

print("H={:.4f}, c={:.4f}".format(H, c))
