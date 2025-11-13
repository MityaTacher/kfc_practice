import numpy as np
from math import gamma as gamma_func
from scipy.special import gammainc
from scipy.stats import gamma as gamma_dist
import pandas
import matplotlib.pyplot as plt

print("Боже, храни нумпай")

# Берем только столбец Time
dataframe = pandas.read_csv("data/123.csv").iloc[:, 1].to_list()
TIME_ARRAY = np.array(dataframe)
TOTAL_LENGHT = len(TIME_ARRAY)
del dataframe

# да начнется пиздец
STEP = 0.05
TOTAL_VALUES = 30
COUNTER = np.arange(TOTAL_VALUES)
LOW = COUNTER * STEP
HIGH = LOW + STEP
LOW[0] = 0.00001  # защита от деления на 0

bins = np.arange(0, TOTAL_VALUES * STEP + STEP, STEP)
ni, edges = np.histogram(TIME_ARRAY, bins=bins)  # делаем данные для построения гистограммы
pi = ni / TOTAL_LENGHT  # векторное деление
cum_pi = np.cumsum(pi)  # накопленная сумма

avg_time = np.average(TIME_ARRAY)
lambda_avg_time = 1 / avg_time

exp_low = 1 - np.exp(-lambda_avg_time * LOW)
exp_high = 1 - np.exp(-lambda_avg_time * HIGH)
exp = exp_high - exp_low  # тут по стандартным формулам

PARETO_A = 0.05
PARETO_K = 1.32

# Параметры для первой гаммы (пик на 0.8)
GAMMA1_A = 6.5
GAMMA1_K = 0.13

# Параметры для второй гаммы (пик на 0.2)
GAMMA2_A = 2.0
GAMMA2_K = 0.1

PARETO_AVG = PARETO_A * PARETO_K / (PARETO_K - 1)
GAMMA1_AVG = GAMMA1_A * GAMMA1_K
GAMMA2_AVG = GAMMA2_A * GAMMA2_K

pareto_low = 1 - (PARETO_A / LOW) ** PARETO_K
pareto_high = 1 - (PARETO_A / HIGH) ** PARETO_K
pareto = pareto_high - pareto_low
pareto[0] = pareto_high[0]


def gamma_cdf(x, a, k):
    """Кумулятивная функция распределения гаммы (CDF)"""
    return gammainc(a, x / k)


# Гамма1 с пиком на 0.8 - ДЛЯ ПЕРВОГО ГРАФИКА
gamma1_high = gamma_cdf(HIGH, GAMMA1_A, GAMMA1_K)
gamma1_low = gamma_cdf(LOW, GAMMA1_A, GAMMA1_K)
gamma1 = gamma1_high - gamma1_low

# Гамма2 с пиком на 0.2 - ДЛЯ ПЕРВОГО ГРАФИКА
gamma2_high = gamma_cdf(HIGH, GAMMA2_A, GAMMA2_K)
gamma2_low = gamma_cdf(LOW, GAMMA2_A, GAMMA2_K)
gamma2 = gamma2_high - gamma2_low


# ДЛЯ ВТОРОГО ГРАФИКА - создаем бимодальное распределение вручную
# Используем нормальные распределения для создания острых пиков
from scipy.stats import norm

# Пик 1 на l = 0.2 с малой дисперсией
PEAK1_MEAN = 0.2
PEAK1_STD = 0.08  # узкий пик

# Пик 2 на l = 0.8 с малой дисперсией
PEAK2_MEAN = 0.8
PEAK2_STD = 0.15  # чуть шире

# Вычисляем вклад каждого пика
peak1 = norm.pdf(LOW, PEAK1_MEAN, PEAK1_STD) * STEP
peak2 = norm.pdf(LOW, PEAK2_MEAN, PEAK2_STD) * STEP

EMPIRICAL_DEVIATION = np.zeros(TOTAL_VALUES)

# Веса для двух пиков
WEIGHT_PEAK1 = 0.6  # 60% на первый пик (0.2)
WEIGHT_PEAK2 = 0.4  # 40% на второй пик (0.8)

# Создаем бимодальное распределение
empirical = peak1 * WEIGHT_PEAK1 + peak2 * WEIGHT_PEAK2 + EMPIRICAL_DEVIATION

# Нормализуем, чтобы сумма равнялась 1
empirical = empirical / np.sum(empirical)

cum_empirical = np.cumsum(empirical)

theor_emp = (empirical - pi) ** 2
cum_theor_emp = (cum_empirical - cum_pi) ** 2

RMSE_PDF = np.average(theor_emp) ** 0.5
RMSE_CDF = np.average(cum_theor_emp) ** 0.5

print(f'RMSE_PDF: {RMSE_PDF}\nRMSE_CDF: {RMSE_CDF}')

# далее чисто графики
LINE_WIDTH = 3
BAR_WIDTH = 0.025

plt.figure(figsize=(13, 5))
plt.gcf().canvas.manager.set_window_title("Rostics")

# Используем LOW для оси X
X_AXIS = LOW

plt.subplot(1, 2, 1)
plt.bar(X_AXIS, pi, label='pi', color='#5b9bd5', width=BAR_WIDTH)
plt.plot(X_AXIS, exp, label="exp", color='#ed7d31', lw=LINE_WIDTH)
plt.plot(X_AXIS, pareto, label="pareto", color='#00b050', lw=LINE_WIDTH)
plt.plot(X_AXIS, gamma1, label="gamma1", color='#ffc000', lw=LINE_WIDTH)
plt.plot(X_AXIS, gamma2, label="gamma2", color='#a020f0', lw=LINE_WIDTH)
plt.grid()
plt.legend()

plt.subplot(1, 2, 2)
plt.bar(X_AXIS, pi, label='pi', color='#5b9bd5', width=BAR_WIDTH)
plt.plot(X_AXIS, empirical, color='black', label='emp', lw=LINE_WIDTH)
plt.grid()
plt.legend()

plt.show()