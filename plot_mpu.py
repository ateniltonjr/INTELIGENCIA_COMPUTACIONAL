import numpy as np
import matplotlib.pyplot as plt
import os

# Caminho absoluto do arquivo de dados, relativo ao script
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'banco_de_dados/teste_motor_fases.csv')
data = np.loadtxt(data_path, delimiter=',')

amostra = data[:, 0]
accel_x = data[:, 1]
accel_y = data[:, 2]
accel_z = data[:, 3]
giro_x = data[:, 4]
giro_y = data[:, 5]
giro_z = data[:, 6]

# Cria vetor de tempo (10000 amostras, 0.1s entre cada)
tempo = amostra * 0.1  # ou np.arange(len(amostra)) * 0.1

fig, axs = plt.subplots(3, 2, figsize=(12, 8))

axs[0, 0].plot(tempo, accel_x, 'b')
axs[0, 0].set_title('Acelerômetro X')
axs[0, 0].set_xlabel('Tempo (s)')
axs[0, 0].set_ylabel('Aceleração X (m/s²)')
axs[0, 0].grid()

axs[1, 0].plot(tempo, accel_y, 'g')
axs[1, 0].set_title('Acelerômetro Y')
axs[1, 0].set_xlabel('Tempo (s)')
axs[1, 0].set_ylabel('Aceleração Y (m/s²)')
axs[1, 0].grid()

axs[2, 0].plot(tempo, accel_z, 'r')
axs[2, 0].set_title('Acelerômetro Z')
axs[2, 0].set_xlabel('Tempo (s)')
axs[2, 0].set_ylabel('Aceleração Z (m/s²)')
axs[2, 0].grid()

axs[0, 1].plot(tempo, giro_x, 'c')
axs[0, 1].set_title('Giroscópio X')
axs[0, 1].set_xlabel('Tempo (s)')
axs[0, 1].set_ylabel('Giro X (°/s)')
axs[0, 1].grid()

axs[1, 1].plot(tempo, giro_y, 'm')
axs[1, 1].set_title('Giroscópio Y')
axs[1, 1].set_xlabel('Tempo (s)')
axs[1, 1].set_ylabel('Giro Y (°/s)')
axs[1, 1].grid()

axs[2, 1].plot(tempo, giro_z, 'y')
axs[2, 1].set_title('Giroscópio Z')
axs[2, 1].set_xlabel('Tempo (s)')
axs[2, 1].set_ylabel('Giro Z (°/s)')
axs[2, 1].grid()

plt.tight_layout()
plt.show()
"""
Adiciona gráficos da FFT dos sinais de acelerômetro e giroscópio
"""

# Função para calcular e plotar FFT
def plot_fft(ax, signal, fs, color, title, ylabel):
    N = len(signal)
    freq = np.fft.rfftfreq(N, d=1/fs)
    fft_vals = np.fft.rfft(signal)
    fft_mag = np.abs(fft_vals)
    ax.plot(freq, fft_mag, color)
    ax.set_title(title)
    ax.set_xlabel('Frequência (Hz)')
    ax.set_ylabel(ylabel)
    ax.grid()

# Frequência de amostragem (1/0.1s = 10Hz)
fs = 10.0

fig_fft, axs_fft = plt.subplots(3, 2, figsize=(12, 8))
plot_fft(axs_fft[0, 0], accel_x, fs, 'b', 'FFT Acelerômetro X', 'Magnitude')
plot_fft(axs_fft[1, 0], accel_y, fs, 'g', 'FFT Acelerômetro Y', 'Magnitude')
plot_fft(axs_fft[2, 0], accel_z, fs, 'r', 'FFT Acelerômetro Z', 'Magnitude')
plot_fft(axs_fft[0, 1], giro_x, fs, 'c', 'FFT Giroscópio X', 'Magnitude')
plot_fft(axs_fft[1, 1], giro_y, fs, 'm', 'FFT Giroscópio Y', 'Magnitude')
plot_fft(axs_fft[2, 1], giro_z, fs, 'y', 'FFT Giroscópio Z', 'Magnitude')

plt.tight_layout()
plt.show()