import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
import argparse

# Установка типа вейвлета для преобразования
WAVELET_TYPE = 'db4' # Добеши 4 - выбран из-за его свойств для анализа сигналов

# Настройка аргументов командной строки для ввода пути к файлу с данными
parser = argparse.ArgumentParser(description='Скрипт обработки сигналов с использованием вейвлет-преобразования')
parser.add_argument('file_path', help='Путь к файлу Excel, содержащему данные сигнала')
args = parser.parse_args()

# Функция для загрузки данных из файла Excel
def load_signal_data(file_path):
    # Попытка загрузить данные из файла Excel
    try:
        data = pd.read_excel(file_path)
        # Преобразование строковых значений в числовые, замена запятых на точки
        md = np.array(data['Ref, m'].apply(lambda x: float(str(x).replace(',', '.'))))
        gr_val = np.array(data['Signal'].apply(lambda x: float(str(x).replace(',', '.'))))
    except ValueError:  # Если столбцы не имеют заголовков
        data = pd.read_excel(file_path, header=None)
        md = np.array(data.iloc[:, 0].apply(lambda x: float(str(x).replace(',', '.'))))
        gr_val = np.array(data.iloc[:, 1].apply(lambda x: float(str(x).replace(',', '.'))))
    return md, gr_val

# Функция для выполнения вейвлет-преобразования и определения интервалов
def wavelet_transform_and_define_intervals(md, gr_val):
    # Выполнение вейвлет-преобразования сигнала
    coeffs = pywt.wavedec(gr_val, WAVELET_TYPE)
    # Определение порогового значения для фильтрации коэффициентов
    threshold = np.sqrt(2 * np.log(len(gr_val))) * np.median(np.abs(coeffs[-1])) / 0.6745
    # Применение порога к коэффициентам
    new_coeffs = list(map(lambda x: pywt.threshold(x, threshold, mode='soft'), coeffs))

    # Определение точек изменения на основе детализированных коэффициентов
    detail_coeffs = new_coeffs[1:]  # Исключение аппроксимационных коэффициентов
    change_points = set()
    max_index = len(md) - 1  # Определение максимально возможного индекса

    for i, level_coeffs in enumerate(detail_coeffs):
        scale_factor = 2 ** i
        significant_points = np.where(np.abs(level_coeffs) > threshold)[0]
        for point in significant_points:
            # Преобразование индекса в реальное положение в исходном сигнале
            actual_point = min(point * scale_factor, max_index)
            change_points.add(actual_point)

    # Сортировка точек изменения для создания интервалов
    intervals = sorted(list(change_points))
    return new_coeffs, intervals

# Функция для восстановления сигнала и его визуализации
def reconstruct_and_visualize(md, gr_val, new_coeffs, intervals):
    # Обратное вейвлет-преобразование для восстановления сигнала
    reconstructed_signal = pywt.waverec(new_coeffs, WAVELET_TYPE)[:len(md)]

    # Построение графиков исходного и восстановленного сигналов
    plt.figure(figsize=(12, 6))
    plt.plot(md, gr_val, label='Исходный сигнал', color='blue')
    plt.plot(md, reconstructed_signal, label='Восстановленный сигнал', linestyle='--', color='red')

    # Отображение интервалов на графике
    for interval in intervals:
        plt.axvline(x=md[interval], color='lightgray', linestyle='dotted', linewidth=0.5)

    plt.title('Сравнение исходного и восстановленного сигналов через вейвлет-преобразование')
    plt.xlabel('Глубина, м')
    plt.ylabel('Значение сигнала')
    plt.legend()
    plt.grid(True)
    plt.show()

# Функция для сохранения интервалов в текстовый файл
def save_intervals_to_txt(intervals, md, file_path):
    # Преобразование индексов интервалов в соответствующие глубины
    interval_values = [md[i] for i in intervals]
    
    # Запись значений интервалов в файл
    with open(file_path, 'w', encoding='utf-8') as file:
        for value in interval_values:
            file.write(f"{value}\n")

# Основной блок скрипта
md, gr_val = load_signal_data(args.file_path)  # Загрузка данных
new_coeffs, intervals = wavelet_transform_and_define_intervals(md, gr_val)  # Выполнение вейвлет-преобразования и определение интервалов
reconstruct_and_visualize(md, gr_val, new_coeffs, intervals)  # Визуализация результатов
save_intervals_to_txt(intervals, md, 'intervals.txt')  # Сохранение интервалов в файл
