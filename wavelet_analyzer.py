import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
import argparse

# Константы для настройки алгоритма
WAVELET_TYPE = 'db4' # Добеши 4 - тип вейвлета для преобразования

# Настройка аргументов командной строки
parser = argparse.ArgumentParser(description='Скрипт обработки сигналов с использованием вейвлет-преобразования')
parser.add_argument('file_path', help='Путь к файлу Excel, содержащему данные сигнала')
args = parser.parse_args()

# Функция для загрузки данных из Excel
def load_signal_data(file_path):
    try:
        data = pd.read_excel(file_path)
        md = np.array(data['Ref, m'].apply(lambda x: float(str(x).replace(',', '.'))))
        gr_val = np.array(data['Signal'].apply(lambda x: float(str(x).replace(',', '.'))))
    except ValueError:
        data = pd.read_excel(file_path, header=None)
        md = np.array(data.iloc[:, 0].apply(lambda x: float(str(x).replace(',', '.'))))
        gr_val = np.array(data.iloc[:, 1].apply(lambda x: float(str(x).replace(',', '.'))))
    return md, gr_val

# Функция для вейвлет-преобразования и восстановления сигнала
def wavelet_transform_and_reconstruct(md, gr_val):
    # Вейвлет-преобразование
    coeffs = pywt.wavedec(gr_val, WAVELET_TYPE)
    
    # Обнуление малозначимых коэффициентов
    threshold = np.sqrt(2 * np.log(len(gr_val))) * np.median(np.abs(coeffs[-1])) / 0.6745
    new_coeffs = list(map(lambda x: pywt.threshold(x, threshold, mode='soft'), coeffs))
    
    # Обратное вейвлет-преобразование
    reconstructed_signal = pywt.waverec(new_coeffs, WAVELET_TYPE)

    return reconstructed_signal[:len(md)]

# Визуализация исходного и восстановленного сигнала
def plot_and_save(md, original_signal, reconstructed_signal):
    plt.figure(figsize=(12, 6))
    plt.plot(md, original_signal, label='Исходный сигнал', color='blue')
    plt.plot(md, reconstructed_signal, label='Восстановленный сигнал', linestyle='--', color='red')
    plt.title('Сравнение исходного и восстановленного сигналов через вейвлет-преобразование')
    plt.xlabel('Глубина, м')
    plt.ylabel('Значение сигнала')
    plt.legend()
    plt.grid(True)
    plt.savefig('wavelet_signal_comparison.png')
    plt.show()

# Функция для сохранения дополнительной информации
def save_wavelet_analysis_info(file_path, original_signal, reconstructed_signal):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write("Разница между исходным и восстановленным сигналом (массив):\n")
        difference_array = original_signal - reconstructed_signal
        file.write(str(difference_array) + "\n\n")

        file.write("Средняя разница между исходным и восстановленным сигналом:\n")
        mean_difference = np.mean(np.abs(difference_array))
        file.write(str(mean_difference) + "\n")

# Основной блок скрипта
md, gr_val = load_signal_data(args.file_path)
reconstructed_signal = wavelet_transform_and_reconstruct(md, gr_val)
plot_and_save(md, gr_val, reconstructed_signal)

# Сохранение дополнительной информации
save_wavelet_analysis_info("wavelet_analysis_info.txt", gr_val, reconstructed_signal)
