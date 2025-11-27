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


# --- 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ ДЛЯ Ya ---
# Читаем весь CSV один раз для формирования Ya (агрегированный ряд)
df = pandas.read_csv("data/123.csv")

# 2-й столбец (index 1) - Время между пакетами
raw_time_deltas = df.iloc[:, 1].to_numpy()
# 6-й столбец (index 5) - Длина пакета
raw_packet_lengths = df.iloc[:, 5].to_numpy()

# Убираем нулевые значения времени (нужно для логарифмов в будущем)
mask = raw_time_deltas != 0
raw_time_deltas = raw_time_deltas[mask]
raw_packet_lengths = raw_packet_lengths[mask]

# Логика агрегации: складываем время пакетов, пока сумма < 2 сек.
# Как только >= 2 сек, считаем среднюю длину пакета.
aggregated_means = []
current_time_sum = 0.0
current_lengths_batch = []
MAX_DURATION = 2.0  # 2 секунды

for t, l in zip(raw_time_deltas, raw_packet_lengths):
    current_time_sum += t
    current_lengths_batch.append(l)

    if current_time_sum >= MAX_DURATION:
        if len(current_lengths_batch) > 0:
            avg_len = np.mean(current_lengths_batch)
            aggregated_means.append(avg_len)
        current_time_sum = 0.0
        current_lengths_batch = []

# Обработка остатка
if len(current_lengths_batch) > 0:
    aggregated_means.append(np.mean(current_lengths_batch))

Y_A = len(aggregated_means)  # Агрегированная длина для формулы WS

# --- 2. ОБРАБОТКА ДАННЫХ ДЛЯ АНАЛИЗА HURST (Ваш исходный код) ---
# Используем raw_time_deltas как базу (это аналог того, что вы читали как iloc[:, 1])
TIME_ARRAY = np.array(raw_time_deltas)

avg_mo, std_mo, avg_var, std_var = nay1(TIME_ARRAY, blocks=20)
deviation_mo = std_mo / avg_mo * 100
deviation_var = std_var / avg_var * 100

TIME_ARRAY = np.diff(np.log10(TIME_ARRAY))  # разница меж соседними логарифмами

# --- Board 1: Variance Method ---
m = [2 ** i for i in range(2, 8)]
m_var = []

for i in m:
    # Защита от выхода за границы массива, если i > len
    if i < len(TIME_ARRAY):
        avg = nay(TIME_ARRAY, rows=i)
        m_var.append(avg)

m_var = np.array(m_var)
m_log = np.log10(m[:len(m_var)])
m_var_log = np.log10(m_var)

coefs = np.polyfit(m_log, m_var_log, 1)  # линейная аппроксимация
k, b = coefs
trend_line = np.polyval(coefs, m_log)

ss_res = np.sum((m_var_log - trend_line) ** 2)
ss_tot = np.sum((m_var_log - np.mean(m_var_log)) ** 2)
r2 = 1 - ss_res / ss_tot

hurst1 = 1 + k / 2
print(f'H1 (Variance): {hurst1}')

# --- Board 2: R/S Analysis ---
hurst2, c, data_rs = hurst.compute_Hc(TIME_ARRAY, kind='change', simplified=True)
print(f"H2 (R/S): {hurst2:.4f}, c={c:.4f}")

# --- 3. ЛОГИКА ПРИНЯТИЯ РЕШЕНИЙ (С ДОСОК) ---
print("-" * 40)
print("DECISION LOGIC:")

ws_factor = 0
case_roman = ""

if hurst1 < 0.5 and hurst2 < 0.5:
    case_roman = "I"
    ws_factor = 0.05
elif hurst1 >= 0.5 and hurst2 >= 0.5:
    case_roman = "II"
    ws_factor = 0.1
elif hurst1 >= 0.5 and hurst2 < 0.5:
    case_roman = "III"
    ws_factor = 0.07
elif hurst1 < 0.5 and hurst2 >= 0.5:
    case_roman = "IV"
    ws_factor = 0.08

calculated_ws = int(ws_factor * Y_A)

print(f"Case: {case_roman}")
print(f"Ya (2s chunks): {Y_A}")
print(f"Recommended Window Size (WS): {calculated_ws}")

# --- 4. ЕДИНЫЙ ГРАФИК (ДВЕ ДОСКИ ВМЕСТЕ) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.canvas.manager.set_window_title(f"Rostics2 - Case {case_roman}")

# График 1 (Variance)
ax1.scatter(m_log, m_var_log, s=50)
ax1.plot(m_log, trend_line, color='red', label=f'y={k:.3f}x+{b:.3f}\nR2={r2:.3f}\nH={hurst1:.3f}')
ax1.set_title("Variance Analysis")
ax1.legend()
ax1.grid()

# График 2 (R/S)
ax2.plot(data_rs[0], c * data_rs[0] ** hurst2, color="deepskyblue", label=f'H={hurst2:.3f}')
ax2.scatter(data_rs[0], data_rs[1], color="purple")
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('Time interval')
ax2.set_ylabel('R/S ratio')
ax2.set_title("R/S Analysis")
ax2.legend()
ax2.grid(True)

# Добавляем инфо-блок на график
plt.figtext(0.02, 0.02,
            f"Case {case_roman}\nYa: {Y_A}\nWS Factor: {ws_factor}\nRec. WS: {calculated_ws}",
            fontsize=10, bbox=dict(facecolor='white', alpha=0.9))

plt.tight_layout()
plt.show()
