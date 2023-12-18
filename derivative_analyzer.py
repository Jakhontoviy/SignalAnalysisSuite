import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Константы для настройки алгоритма
SENSITIVITY = 0.7   # Чувствительность алгоритма
PERCENTILE = 1      # Процентиль для определения пороговых значений производной
H_MIN = 0.4         # Минимальное расстояние между границами

# Настройка аргументов командной строки для скрипта
parser = argparse.ArgumentParser(description='Скрипт обработки сигналов')
parser.add_argument('file_path', help='Путь к файлу Excel, содержащему данные сигнала')
args = parser.parse_args()

# Функция для загрузки и обработки данных из Excel
def load_signal_data(file_path):
    try:
        # Попытка чтения данных с предположением, что заголовки присутствуют
        data = pd.read_excel(file_path)
        md = np.array(data['Ref, m'].apply(lambda x: float(str(x).replace(',', '.'))))
        gr_val = np.array(data['Signal'].apply(lambda x: float(str(x).replace(',', '.'))))
    except ValueError:
        # Чтение данных без заголовков, если предыдущая попытка привела к ошибке
        data = pd.read_excel(file_path, header=None)
        md = np.array(data.iloc[:, 0].apply(lambda x: float(str(x).replace(',', '.'))))
        gr_val = np.array(data.iloc[:, 1].apply(lambda x: float(str(x).replace(',', '.'))))
    
    return md, gr_val

# Функция обработки сигнала
def process_signal(md, gr_val):
    # Вычисление производной сигнала
    sr = (md.max() - md.min()) / len(md)
    derivative = np.gradient(gr_val, sr)
    
    # Определение пороговых значений для производной
    dmax = np.percentile(derivative, 100 - PERCENTILE)
    dmin = np.percentile(derivative, PERCENTILE)
    sens_max = dmax * (1 - SENSITIVITY)
    sens_min = abs(dmin * (1 - SENSITIVITY))
    
    # Определение границ интервалов
    boundaries = []
    for i in range(1, len(md)):
        if (derivative[i-1] < sens_max and derivative[i] >= sens_max) or \
           (derivative[i-1] > sens_min and derivative[i] <= sens_min):
            if len(boundaries) == 0 or (md[i] - boundaries[-1]) >= H_MIN:
                boundaries.append(md[i])
    
    # Вычисление средних значений в каждом интервале
    average_values = []
    start_index = 0
    for boundary in boundaries:
        end_index = np.searchsorted(md, boundary)
        average_values.append(np.mean(gr_val[start_index:end_index]))
        start_index = end_index
    
    # Восстановление сигнала из средних значений
    reconstructed_signal = np.interp(md, [md[0]] + boundaries, [gr_val[0]] + average_values)
    
    return boundaries, average_values, reconstructed_signal

# Функция для отображения и сохранения графика
def plot_and_save(md, gr_val, reconstructed_signal):
    plt.figure(figsize=(12, 6))
    plt.plot(md, gr_val, label='Исходный сигнал', color='blue')
    plt.plot(md, reconstructed_signal, label='Восстановленный сигнал', linestyle='--', color='red')
    plt.title('Сравнение исходного и восстановленного сигналов')
    plt.xlabel('Глубина, м')
    plt.ylabel('Значение сигнала')
    plt.legend()
    plt.grid(True)
    plt.savefig('signal_comparison.png')
    plt.show()

# Функция для сохранения дополнительной информации
def save_additional_info(file_path, boundaries, average_values, reconstructed_signal, original_signal):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write("Найденные границы:\n")
        file.write(str(boundaries) + "\n\n")

        file.write("Осредненные значения сигнала в каждом интервале:\n")
        file.write(str(average_values) + "\n\n")

        file.write("Разница между исходным и восстановленным сигналом (массив):\n")
        difference_array = original_signal - reconstructed_signal
        file.write(str(difference_array) + "\n\n")

        file.write("Средняя разница между исходным и восстановленным сигналом:\n")
        mean_difference = np.mean(np.abs(difference_array))
        file.write(str(mean_difference) + "\n")

# Основной блок скрипта
md, gr_val = load_signal_data(args.file_path)
boundaries, average_values, reconstructed_signal = process_signal(md, gr_val)
plot_and_save(md, gr_val, reconstructed_signal)

# Сохранение дополнительной информации
save_additional_info("additional_info.txt", boundaries, average_values, reconstructed_signal, gr_val)