import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Константы для настройки алгоритма
NUM_HARMONICS = 3000    # Выбор количества гармоник для анализа - значение можно оптимизировать

# Настройка аргументов командной строки
parser = argparse.ArgumentParser(description='Скрипт обработки сигналов с использованием преобразования Фурье')
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

# Функция преобразования Фурье и восстановления сигнала
def fourier_transform_and_reconstruct(md, gr_val, num_harmonics):
    # Преобразование Фурье
    fft_result = np.fft.fft(gr_val)
    frequencies = np.fft.fftfreq(len(gr_val), d=(md[1] - md[0]))

    # Отбор num_harmonics наиболее значимых гармоник
    indices = np.argsort(np.abs(fft_result))[-num_harmonics:]

    # Обнуление всех остальных гармоник
    fft_result_filtered = np.zeros_like(fft_result)
    fft_result_filtered[indices] = fft_result[indices]

    # Обратное преобразование Фурье
    reconstructed_signal = np.fft.ifft(fft_result_filtered)

    return reconstructed_signal.real

# Визуализация исходного и восстановленного сигнала
def plot_and_save(md, original_signal, reconstructed_signal):
    plt.figure(figsize=(12, 6))
    plt.plot(md, original_signal, label='Исходный сигнал', color='blue')
    plt.plot(md, reconstructed_signal, label='Восстановленный сигнал', linestyle='--', color='red')
    plt.title('Сравнение исходного и восстановленного сигналов через преобразование Фурье')
    plt.xlabel('Глубина, м')
    plt.ylabel('Значение сигнала')
    plt.legend()
    plt.grid(True)
    plt.savefig('fourier_signal_comparison.png')
    plt.show()

# Функция для сохранения дополнительной информации
def save_fourier_analysis_info(file_path, original_signal, reconstructed_signal, num_harmonics):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write("Количество использованных гармоник: {}\n\n".format(num_harmonics))

        file.write("Разница между исходным и восстановленным сигналом (массив):\n")
        difference_array = original_signal - reconstructed_signal
        file.write(str(difference_array) + "\n\n")

        file.write("Средняя разница между исходным и восстановленным сигналом:\n")
        mean_difference = np.mean(np.abs(difference_array))
        file.write(str(mean_difference) + "\n")

# Основной блок скрипта
md, gr_val = load_signal_data(args.file_path)
reconstructed_signal = fourier_transform_and_reconstruct(md, gr_val, NUM_HARMONICS)
plot_and_save(md, gr_val, reconstructed_signal)

# Сохранение дополнительной информации
save_fourier_analysis_info("fourier_analysis_info.txt", gr_val, reconstructed_signal, NUM_HARMONICS)