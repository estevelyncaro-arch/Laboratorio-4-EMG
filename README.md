# Laboratorio-4-EMG
## Resumen
En el desarrollo de este laboratorio se busca analizar señales electromiográficas (EMG) emuladas por el generador de señales y señales reales, realizando una comparación de el comportamiento de cada una. En la señal de EMG real se busca detectar la fatiga muscular capturando la señal en tiempo real y aplicando filtros para elimminar el ruido para un mejor analisis de las contracciónes.
## Parte A 
![Diagramas lab 4_page-0001](https://github.com/user-attachments/assets/fac9d933-47e5-4971-a120-a657c7122291)

Para esta primera sección se realiza la captura de una señal electromiográfica (EMG) emulada por el generador de señales con ayuda de un DAQ, se simulan 5 contracciones. Una vez adquirida la señal se importa a python y se grafica implementando el siguiente código:

```python
import numpy as np
import matplotlib.pyplot as plt

# Cargar el archivo (2 filas: tiempo y voltaje)
data = np.loadtxt("labo fs200.txt")

# Separar filas
tiempo = data[0, :]   # primera fila
senal  = data[1, :]   # segunda fila

# Graficar
plt.figure(figsize=(9, 4))
plt.plot(tiempo, senal, linewidth=1)
plt.title("Señal adquirida en Dev5/ai0")
plt.xlabel("Tiempo [s]")
plt.ylabel("Voltaje [V]")
plt.grid(True)
plt.tight_layout()
plt.show()
```
Obteniendo el siguiente gráfico:

<img width="889" height="390" alt="image" src="https://github.com/user-attachments/assets/1cc61e34-39ba-43b6-be91-cef91040399d" />

Luego se segmento la señal capturada en 5 partes, con el siguiente codigo:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

data = np.loadtxt("/labo fs200.txt")
t = data[0, :]
x = data[1, :]
fs = 200  # Frecuencia de muestreo [Hz]

x_rect = np.abs(x - np.mean(x))
b, a = butter(2, 2/(fs/2), btype='low')
envolvente = filtfilt(b, a, x_rect)
envolvente_norm = envolvente / np.max(envolvente)

umbral = np.mean(envolvente_norm) + 0.5*np.std(envolvente_norm)
activa = envolvente_norm > umbral
start_idx = np.where(np.diff(activa.astype(int)) == 1)[0]
end_idx = np.where(np.diff(activa.astype(int)) == -1)[0]

if len(end_idx) > 0 and end_idx[0] < start_idx[0]:
    end_idx = end_idx[1:]
if len(start_idx) > len(end_idx):
    start_idx = start_idx[:-1]

pre_ext = int(0.15 * fs)
post_ext = int(0.10 * fs)
start_idx_adj = np.clip(start_idx - pre_ext, 0, len(x)-1)
end_idx_adj   = np.clip(end_idx + post_ext, 0, len(x)-1)

contracciones = []  # Lista para almacenar las contracciones

for i, (ini, fin) in enumerate(zip(start_idx_adj, end_idx_adj)):
    seg_t = t[ini:fin]
    seg_x = x[ini:fin]
    contracciones.append({
        "tiempo": seg_t,
        "senal": seg_x,
        "indice_inicio": ini,
        "indice_fin": fin
    })

    # Graficar cada una
    plt.figure(figsize=(6,2))
    plt.plot(seg_t, seg_x)
    plt.title(f"Contracción {i+1}")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Voltaje [V]")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plt.plot(t, x/np.max(np.abs(x)), label="Señal filtrada", color='blue', alpha=0.7)
plt.plot(t, envolvente_norm, label="Energía normalizada", color='orange', linewidth=2)
for i in range(len(start_idx_adj)):
    plt.axvspan(t[start_idx_adj[i]], t[end_idx_adj[i]], color='red', alpha=0.3)

plt.title("Segmentación automática de contracciones musculares (ajustada)")
plt.xlabel("Tiempo [s]")
plt.ylabel("Voltaje [V]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```
evidenciandolo así en las siguientes imagenes:

<img width="590" height="190" alt="image" src="https://github.com/user-attachments/assets/158493f9-7b90-4674-8bed-e65e4c01f29e" />
<img width="590" height="190" alt="image" src="https://github.com/user-attachments/assets/5a1c6de1-dad5-40d8-bcd5-a9a98cc1b50a" />
<img width="590" height="190" alt="image" src="https://github.com/user-attachments/assets/ff3ad7a4-8350-4dfe-afb0-b6c4ff9dbe58" />
<img width="590" height="190" alt="image" src="https://github.com/user-attachments/assets/0139f105-9cda-4814-8caf-339d8a645a83" />
<img width="590" height="190" alt="image" src="https://github.com/user-attachments/assets/cce05647-3072-4066-92fb-37b1b0e7cc31" />
<img width="989" height="390" alt="image" src="https://github.com/user-attachments/assets/dbf849fa-8c45-4efa-b116-15a21fda0673" />

Para así tomar la frecuencia media y la frecuencia mediana con el siguiente codigo:

```python
import pandas as pd
import matplotlib.pyplot as plt

df_resultados = pd.DataFrame(resultados)
df_resultados.columns = ["Contracción", "Frecuencia media (Hz)", "Frecuencia mediana (Hz)"]

# Mostrar tabla
print("\n=== TABLA DE RESULTADOS ===\n")
print(df_resultados.to_string(index=False))

plt.figure(figsize=(8,4))
plt.plot(df_resultados["Contracción"], df_resultados["Frecuencia media (Hz)"], marker='o', label="Frecuencia media", color='steelblue')
plt.plot(df_resultados["Contracción"], df_resultados["Frecuencia mediana (Hz)"], marker='s', label="Frecuencia mediana", color='orange')

plt.title("Evolución de las frecuencias por contracción")
plt.xlabel("Número de contracción")
plt.ylabel("Frecuencia [Hz]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```
obteniendo una tabla y la siguiente grafica:


#### TABLA DE RESULTADOS 
| Contracción | Frecuencia media (Hz) | Frecuencia mediana (Hz) |
|--------------|-----------------------|--------------------------|
| 1            | 534.134160            | 283.018868              |
| 2            | 524.413970            | 283.018868              |
| 3            | 546.472563            | 285.714286              |
| 4            | 543.589980            | 280.373832              |
| 5            | 540.716311            | 280.373832              |


<img width="790" height="390" alt="image" src="https://github.com/user-attachments/assets/4ccb0ebd-9880-427b-9e49-8539cd6fd7ad" />

Durante la serie de contracciones simuladas, tanto la frecuencia media como la mediana muestran variaciones leves, sin una disminución sostenida. Esto indica que la señal muscular se mantiene estable, sin evidencia de fatiga progresiva a lo largo del ejercicio

## Parte B

![Diagramas lab 4_page-0002](https://github.com/user-attachments/assets/4fbf2236-846e-4194-8106-702e431986bd)

En esta segunda parte se realizo la captura de las contracciones a tiepo real con ayuda de un modulo AD8232 y con electrodos, estoss se conectan al ante brazo y con la DAQ se realiza la captura. 

![Imagen de WhatsApp 2025-10-23 a las 09 32 57_ef6c2349](https://github.com/user-attachments/assets/abfbc1ac-84cf-4a14-a30c-59f939104f1c)

con ayuda del siguiente codigo se pudo hacer la captura:

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import nidaqmx
from nidaqmx.constants import AcquisitionType
from threading import Thread, Event
from collections import deque
import datetime
import time

fs = 10000            # Frecuencia de muestreo (Hz)
canal = "Dev5/ai0"    # Cambia según tu dispositivo
tamano_bloque = int(fs * 0.05)  # 50 ms por bloque
ventana_tiempo = 3.0             # segundos visibles en la gráfica

# Buffers
buffer_graf = deque(maxlen=int(fs * ventana_tiempo))  # solo últimos 3 s
datos_guardados = []  # toda la señal

# Control de hilos
adquiriendo = Event()
detener_hilo = Event()
thread_lectura = None


def hilo_lectura():
    """Lee continuamente datos del DAQ en un hilo aparte."""
    global datos_guardados, buffer_graf
    task = nidaqmx.Task()
    task.ai_channels.add_ai_voltage_chan(canal)
    task.timing.cfg_samp_clk_timing(rate=fs, sample_mode=AcquisitionType.CONTINUOUS)
    task.start()
    print(f"\n▶ Adquisición iniciada en {canal} ({fs} Hz).")

    while not detener_hilo.is_set():
        if adquiriendo.is_set():
            try:
                datos = task.read(number_of_samples_per_channel=tamano_bloque)
                buffer_graf.extend(datos)
                datos_guardados.extend(datos)
            except Exception as e:
                print("⚠ Error de lectura:", e)
                break
        else:
            time.sleep(0.05)

    task.stop()
    task.close()
    print("⏹ Adquisición detenida correctamente.")


def iniciar(event):
    global thread_lectura
    if not adquiriendo.is_set():
        if thread_lectura is None or not thread_lectura.is_alive():
            detener_hilo.clear()
            thread_lectura = Thread(target=hilo_lectura, daemon=True)
            thread_lectura.start()
        adquiriendo.set()
        print("▶ Grabando...")

def detener(event):
    """Detiene y guarda los datos."""
    adquiriendo.clear()
    detener_hilo.set()
    time.sleep(0.3)

    if datos_guardados:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = f"senal_EMG_{timestamp}.txt"
        tiempos = np.arange(len(datos_guardados)) / fs
        data = np.column_stack((tiempos, datos_guardados))
        np.savetxt(nombre_archivo, data, fmt="%.6f", header="Tiempo(s)\tVoltaje(V)")
        print(f"✅ Señal guardada en {nombre_archivo} ({len(datos_guardados)} muestras)")
    else:
        print("⚠ No se capturaron datos.")


fig, ax = plt.subplots(figsize=(10, 4))
plt.subplots_adjust(bottom=0.25)
linea, = ax.plot([], [], lw=1.2, color='royalblue')
ax.set_xlim(0, ventana_tiempo)
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel("Tiempo [s]")
ax.set_ylabel("Voltaje [V]")
ax.set_title("Señal EMG continua en tiempo real")
ax.grid(True, linestyle="--", alpha=0.6)

x = np.linspace(0, ventana_tiempo, int(fs * ventana_tiempo))
y = np.zeros_like(x)

def actualizar(frame):
    if len(buffer_graf) > 0:
        y = np.array(buffer_graf)
        if len(y) < len(x):
            y = np.pad(y, (len(x)-len(y), 0), constant_values=0)
        linea.set_data(x, y)
    return linea,

ax_iniciar = plt.axes([0.3, 0.1, 0.15, 0.075])
ax_detener = plt.axes([0.55, 0.1, 0.2, 0.075])
btn_iniciar = Button(ax_iniciar, 'Iniciar', color='lightgreen', hovercolor='green')
btn_detener = Button(ax_detener, 'Detener y Guardar', color='lightcoral', hovercolor='red')
btn_iniciar.on_clicked(iniciar)
btn_detener.on_clicked(detener)

from matplotlib.animation import FuncAnimation
ani = FuncAnimation(fig, actualizar, interval=50, blit=True)
plt.tight_layout()
plt.show()
```

Luego se pasaraon los datos para graficarlos y se tomaron los primeros 10 segundos y los ultimos 10 para poder evidenciar la fatiga en la siguiente grafica con ayuda del cofigo:

```python
import numpy as np
import matplotlib.pyplot as plt

ruta_txt = "/senal_EMG_captura_2.txt"   # Cambia por el nombre de tu archivo
col_tiempo = 0            # índice de la columna de tiempo
col_voltaje = 1           # índice de la columna de voltaje


# Carga el archivo ignorando líneas vacías o comentarios
datos = np.loadtxt(ruta_txt)

# Separa las columnas
tiempo = datos[:, col_tiempo]
voltaje = datos[:, col_voltaje]


mascara_inicio = tiempo <= 10
t_inicio = tiempo[mascara_inicio]
v_inicio = voltaje[mascara_inicio]

t_final_max = tiempo.max()
mascara_final = tiempo >= (t_final_max - 10)
t_final = tiempo[mascara_final]
v_final = voltaje[mascara_final]


plt.figure(figsize=(12,6))

plt.subplot(2,1,1)
plt.plot(t_inicio, v_inicio, color='b')
plt.title("Primeros 10 segundos")
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje (V)")
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(t_final, v_final, color='r')
plt.title("Últimos 10 segundos")
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje (V)")
plt.grid(True)

plt.tight_layout()
plt.show()
```

<img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/72288403-010d-46ac-a253-9b0bcf061ecd" />

Despues se aplico un filtro pasabanda (20–450 Hz) para eliminar ruido y artefactos.
esto se logro con el siguiente codigo:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


ruta_txt = "/senal_EMG_captura_2.txt"   # Cambia por tu archivo
col_tiempo = 0
col_voltaje = 1
fs = 1000                # Frecuencia de muestreo (Hz) — cámbiala según tu caso


datos = np.loadtxt(ruta_txt)
tiempo = datos[:, col_tiempo]
voltaje = datos[:, col_voltaje]


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def aplicar_filtro(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Aplicar filtro 20–450 Hz
voltaje_filtrado = aplicar_filtro(voltaje, 20, 450, fs, order=4)

mascara_inicio = tiempo <= 10
t_inicio = tiempo[mascara_inicio]
v_inicio = voltaje_filtrado[mascara_inicio]

t_final_max = tiempo.max()
mascara_final = tiempo >= (t_final_max - 20)
t_final = tiempo[mascara_final]
v_final = voltaje_filtrado[mascara_final]

plt.figure(figsize=(12,6))

plt.subplot(2,1,1)
plt.plot(t_inicio, v_inicio, color='b')
plt.title("Primeros 10 segundos (filtrados 20–450 Hz)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje (V)")
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(t_final, v_final, color='r')
plt.title("Últimos 20 segundos (filtrados 20–450 Hz)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje (V)")
plt.grid(True)

plt.tight_layout()
plt.show()
```

evidenciando así la grafica:

<img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/88b76770-7823-4a2e-a7e5-fadd8a778e55" />

El método que se utilizo para seccionar la señal se llama adaptive statistical threshold la cual no usa cruces por cero ni transformadas de frecuencia, sino que se basa en la amplitud de la envolvente y un umbral estadístico dinámico para detectar las fases activas del músculo.  
gracias a esto se pudo halla las contarcciones por segundo

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

fs = 1000  # Frecuencia de muestreo [Hz]
ruta_txt = "/senal_EMG_captura_2.txt"   # <-- cambia esta ruta

data = np.loadtxt(ruta_txt)
t = data[:, 0]
x = data[:, 1]

lowcut, highcut, orden = 20, 450, 4

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return b, a

def aplicar_filtro(x, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, x)

x_f = aplicar_filtro(x, lowcut, highcut, fs, orden)

x_rect = np.abs(x_f - np.mean(x_f))
b, a = butter(2, 2/(fs/2), btype='low')
env = filtfilt(b, a, x_rect)
env /= np.max(env)

factor_umbral = 1.2       # más alto = menos detecciones
umbral = np.mean(env) + factor_umbral * np.std(env)
activa = env > umbral

# bordes
start_idx = np.where(np.diff(activa.astype(int)) == 1)[0]
end_idx   = np.where(np.diff(activa.astype(int)) == -1)[0]
if len(end_idx) > 0 and end_idx[0] < start_idx[0]:
    end_idx = end_idx[1:]
if len(start_idx) > len(end_idx):
    start_idx = start_idx[:-1]

# extensiones
pre_ext  = int(0.05 * fs)
post_ext = int(0.05 * fs)
start_idx = np.clip(start_idx - pre_ext, 0, len(x)-1)
end_idx   = np.clip(end_idx + post_ext, 0, len(x)-1)

# eliminar eventos cortos
min_duracion = int(0.15 * fs)    # 150 ms
contracciones = [(i, f) for i, f in zip(start_idx, end_idx) if (f - i) > min_duracion]

# fusionar eventos cercanos
fusionadas = []
if contracciones:
    ini, fin = contracciones[0]
    for i, f in contracciones[1:]:
        if i - fin < 0.30 * fs:       # < 300 ms ⇒ misma contracción
            fin = f
        else:
            fusionadas.append((ini, fin))
            ini, fin = i, f
    fusionadas.append((ini, fin))

segmentos = {}
for k, (ini, fin) in enumerate(fusionadas, 1):
    nombre = f"c{k}"
    segmentos[nombre] = x_f[ini:fin]
    globals()[nombre] = segmentos[nombre]
    print(f"Contracción {k} guardada como '{nombre}' ({t[ini]:.2f}s – {t[fin]:.2f}s)")

print(f"\n🔹 Total detectadas: {len(fusionadas)}")


plt.figure(figsize=(10,4))
plt.plot(t, x_f/np.max(np.abs(x_f)), color='blue', alpha=0.7, label="Señal filtrada (20–450 Hz)")
plt.plot(t, env, color='orange', lw=2, label="Envolvente normalizada")
plt.axhline(umbral, color='red', ls='--', label=f"Umbral ({umbral:.2f})")

for ini, fin in fusionadas:
    plt.axvspan(t[ini], t[fin], color='red', alpha=0.25)

plt.title(f"Detección automática de contracciones ({len(fusionadas)} encontradas)")
plt.xlabel("Tiempo [s]")
plt.ylabel("Voltaje [V]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```
obteniendo asi los siguiente resultados:

 Contracción 1 guardada como 'c1' (0.15s – 0.28s)
 
Contracción 2 guardada como 'c2' (0.74s – 0.86s)

Contracción 3 guardada como 'c3' (1.30s – 1.42s)

Contracción 4 guardada como 'c4' (1.84s – 1.94s)

Contracción 5 guardada como 'c5' (2.48s – 2.57s)

Contracción 6 guardada como 'c6' (3.10s – 3.18s)

Contracción 7 guardada como 'c7' (3.61s – 3.70s)

Contracción 8 guardada como 'c8' (4.17s – 4.19s)

Contracción 9 guardada como 'c9' (4.23s – 4.32s)

Contracción 10 guardada como 'c10' (4.82s – 4.86s)

Contracción 11 guardada como 'c11' (5.98s – 6.13s)

Contracción 12 guardada como 'c12' (6.61s – 6.62s)

Contracción 13 guardada como 'c13' (7.05s – 7.08s)

Contracción 14 guardada como 'c14' (7.39s – 7.43s)

Contracción 15 guardada como 'c15' (8.01s – 8.03s)

Contracción 16 guardada como 'c16' (8.07s – 8.11s)

Contracción 17 guardada como 'c17' (8.56s – 8.67s)

Contracción 18 guardada como 'c18' (9.13s – 9.18s)

Contracción 19 guardada como 'c19' (9.22s – 9.26s)

Contracción 20 guardada como 'c20' (9.68s – 9.79s)

Contracción 21 guardada como 'c21' (10.21s – 10.27s)

Contracción 22 guardada como 'c22' (10.32s – 10.37s)

Contracción 23 guardada como 'c23' (10.83s – 10.93s)

Contracción 24 guardada como 'c24' (11.44s – 11.52s)

Contracción 25 guardada como 'c25' (12.02s – 12.12s)

Contracción 26 guardada como 'c26' (12.58s – 12.68s)

Contracción 27 guardada como 'c27' (13.14s – 13.25s)

Contracción 28 guardada como 'c28' (13.64s – 13.73s)

Contracción 29 guardada como 'c29' (14.28s – 14.38s)

Contracción 30 guardada como 'c30' (14.77s – 14.90s)

Contracción 31 guardada como 'c31' (15.36s – 15.37s)

Contracción 32 guardada como 'c32' (15.43s – 15.53s)

Contracción 33 guardada como 'c33' (15.90s – 16.02s)

Contracción 34 guardada como 'c34' (16.53s – 16.59s)

Contracción 35 guardada como 'c35' (17.02s – 17.15s)

Contracción 36 guardada como 'c36' (17.75s – 17.80s)

Contracción 37 guardada como 'c37' (18.34s – 18.38s)

Contracción 38 guardada como 'c38' (18.86s – 18.90s)

Contracción 39 guardada como 'c39' (19.41s – 19.50s)

Contracción 40 guardada como 'c40' (20.61s – 20.69s)

Contracción 41 guardada como 'c41' (21.19s – 21.26s)

Contracción 42 guardada como 'c42' (21.89s – 21.92s)

Contracción 43 guardada como 'c43' (22.34s – 22.44s)

Contracción 44 guardada como 'c44' (22.48s – 22.54s)

Contracción 45 guardada como 'c45' (22.92s – 22.96s)

Contracción 46 guardada como 'c46' (23.04s – 23.08s)

Contracción 47 guardada como 'c47' (23.69s – 23.72s)

Contracción 48 guardada como 'c48' (24.81s – 24.84s)

Contracción 49 guardada como 'c49' (25.35s – 25.44s)

Contracción 50 guardada como 'c50' (25.94s – 25.97s)

Contracción 51 guardada como 'c51' (26.53s – 26.61s)

Contracción 52 guardada como 'c52' (27.15s – 27.26s)

Contracción 53 guardada como 'c53' (27.73s – 27.86s)

Contracción 54 guardada como 'c54' (28.27s – 28.44s)

Contracción 55 guardada como 'c55' (28.97s – 29.03s)

Contracción 56 guardada como 'c56' (29.51s – 29.56s)

Contracción 57 guardada como 'c57' (29.60s – 29.63s)

Contracción 58 guardada como 'c58' (30.13s – 30.23s)

Contracción 59 guardada como 'c59' (30.86s – 30.90s)

Contracción 60 guardada como 'c60' (31.39s – 31.43s)

Contracción 61 guardada como 'c61' (32.57s – 32.66s)

Contracción 62 guardada como 'c62' (33.22s – 33.26s)

Contracción 63 guardada como 'c63' (35.57s – 35.78s)

Contracción 64 guardada como 'c64' (36.23s – 36.38s)

Contracción 65 guardada como 'c65' (36.80s – 36.95s)

Contracción 66 guardada como 'c66' (37.45s – 37.51s)

Contracción 67 guardada como 'c67' (38.02s – 38.08s)

Contracción 68 guardada como 'c68' (38.12s – 38.20s)

Contracción 69 guardada como 'c69' (38.66s – 38.83s)

Contracción 70 guardada como 'c70' (39.27s – 39.32s)

Contracción 71 guardada como 'c71' (39.36s – 39.43s)

Contracción 72 guardada como 'c72' (39.93s – 40.04s)

Contracción 73 guardada como 'c73' (40.50s – 40.55s)

Contracción 74 guardada como 'c74' (41.06s – 41.10s)

Contracción 75 guardada como 'c75' (41.81s – 41.85s)

Contracción 76 guardada como 'c76' (42.30s – 42.47s)

Contracción 77 guardada como 'c77' (43.00s – 43.06s)

Contracción 78 guardada como 'c78' (43.54s – 43.59s)

Contracción 79 guardada como 'c79' (43.63s – 43.69s)

Contracción 80 guardada como 'c80' (44.31s – 44.34s)

Contracción 81 guardada como 'c81' (44.38s – 44.42s)

Contracción 82 guardada como 'c82' (44.95s – 45.00s)

Contracción 83 guardada como 'c83' (45.62s – 45.86s)

Contracción 84 guardada como 'c84' (46.40s – 46.44s)

Contracción 85 guardada como 'c85' (47.02s – 47.06s)

Contracción 86 guardada como 'c86' (47.56s – 47.68s)

Contracción 87 guardada como 'c87' (48.90s – 48.93s)

Contracción 88 guardada como 'c88' (50.19s – 50.21s)

Contracción 89 guardada como 'c89' (50.73s – 50.81s)

Contracción 90 guardada como 'c90' (51.38s – 51.48s)

Contracción 91 guardada como 'c91' (52.03s – 52.06s)

Contracción 92 guardada como 'c92' (53.84s – 53.90s)

Contracción 93 guardada como 'c93' (53.93s – 53.97s)

Contracción 94 guardada como 'c94' (54.42s – 54.48s)

Contracción 95 guardada como 'c95' (55.08s – 55.19s)

Contracción 96 guardada como 'c96' (55.71s – 55.86s)

Contracción 97 guardada como 'c97' (56.40s – 56.53s)

Contracción 98 guardada como 'c98' (57.09s – 57.14s)

Contracción 99 guardada como 'c99' (57.73s – 57.79s)

Contracción 100 guardada como 'c100' (58.20s – 58.26s)

Contracción 101 guardada como 'c101' (58.34s – 58.38s)

Contracción 102 guardada como 'c102' (58.91s – 59.01s)

Contracción 103 guardada como 'c103' (59.70s – 59.76s)

Contracción 104 guardada como 'c104' (59.81s – 59.83s)

Contracción 105 guardada como 'c105' (60.27s – 60.43s)

Contracción 106 guardada como 'c106' (60.92s – 60.98s)

Contracción 107 guardada como 'c107' (61.08s – 61.11s)

Contracción 108 guardada como 'c108' (61.56s – 61.60s)

Contracción 109 guardada como 'c109' (61.73s – 61.77s)

Contracción 110 guardada como 'c110' (62.35s – 62.43s)

🔹 Total detectadas: 110

<img width="989" height="390" alt="image" src="https://github.com/user-attachments/assets/4f0af3d9-4472-4b4a-9c39-f4a1aa4ddab2" />

En la siguiente parte se calcularon la frecuencia media y la frecuencia mediana con el siguiente codigo:

```python
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq

fs = 1000  # Frecuencia de muestreo [Hz]
num_contracciones = 110  # número total de contracciones (c1, c2, ..., c110)

resultados = []

for i in range(1, num_contracciones + 1):
    var_name = f"c{i}"
    if var_name in globals():
        signal = globals()[var_name]
        N = len(signal)
        duracion = N / fs

        # FFT
        yf = np.abs(fft(signal))
        xf = fftfreq(N, 1/fs)

        # Solo frecuencias positivas
        pos_mask = xf > 0
        xf = xf[pos_mask]
        yf = yf[pos_mask]

        # Calcular frecuencia media y mediana (ponderadas por amplitud)
        f_media = np.sum(xf * yf) / np.sum(yf)
        f_cum = np.cumsum(yf) / np.sum(yf)
        f_mediana = xf[np.where(f_cum >= 0.5)[0][0]]

        resultados.append({
            "Contracción": var_name,
            "Duración (s)": round(duracion, 3),
            "Frecuencia Media (Hz)": round(f_media, 2),
            "Frecuencia Mediana (Hz)": round(f_mediana, 2)
        })

tabla = pd.DataFrame(resultados)
print(tabla)

tabla.to_csv("resumen_contracciones.csv", index=False)
print("\n✅ Tabla guardada como 'resumen_contracciones.csv'")
```
para evidenciar la siguiente tabla:

| Contracción | Duración (s) | Frecuencia Media (Hz) | Frecuencia Mediana (Hz) |
|--------------|--------------|-----------------------|--------------------------|
| c1  | 1.307 | 63.24 | 25.25 |
| c2  | 1.282 | 52.76 | 28.08 |
| c3  | 1.200 | 53.12 | 26.67 |
| c4  | 0.997 | 49.89 | 25.08 |
| c5  | 0.886 | 51.71 | 23.70 |
| ... | ...   | ...   | ...   |
| c106 | 0.528 | 48.04 | 30.30 |
| c107 | 0.309 | 49.62 | 32.36 |
| c108 | 0.410 | 47.86 | 29.27 |
| c109 | 0.403 | 60.58 | 34.74 |
| c110 | 0.809 | 59.43 | 33.37 |

Ahora se evidenciar las frecuencias para la fatiga 

```python
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np

df = pd.read_csv("resumen_contracciones.csv")

# Normalizar nombres de columnas si tienen espacios
df.columns = [col.strip().replace(" ", "_") for col in df.columns]

# Agregar un índice de contracción (orden temporal)
df["N°"] = range(1, len(df) + 1)


plt.figure(figsize=(10,6))
plt.plot(df["N°"], df["Frecuencia_Media_(Hz)"], 'o-', color='royalblue', label="Frecuencia Media (Hz)")
plt.plot(df["N°"], df["Frecuencia_Mediana_(Hz)"], 'o-', color='orange', label="Frecuencia Mediana (Hz)")

# Calcular líneas de tendencia
slope_media, intercept_media, *_ = linregress(df["N°"], df["Frecuencia_Media_(Hz)"])
slope_mediana, intercept_mediana, *_ = linregress(df["N°"], df["Frecuencia_Mediana_(Hz)"])

tendencia_media = intercept_media + slope_media * np.array(df["N°"])
tendencia_mediana = intercept_mediana + slope_mediana * np.array(df["N°"])

plt.plot(df["N°"], tendencia_media, '--', color='blue', alpha=0.7, label="Tendencia Media")
plt.plot(df["N°"], tendencia_mediana, '--', color='red', alpha=0.7, label="Tendencia Mediana")

plt.title("Evolución de la Frecuencia Media y Mediana durante la Fatiga Muscular")
plt.xlabel("Número de Contracción")
plt.ylabel("Frecuencia (Hz)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("🔹 Pendiente de la Frecuencia Media:", round(slope_media, 4))
print("🔹 Pendiente de la Frecuencia Mediana:", round(slope_mediana, 4))

if slope_media < 0 and slope_mediana < 0:
    print("\n📉 Las dos pendientes son negativas → tendencia descendente clara.")
    print("👉 Esto indica la aparición de fatiga muscular progresiva.")
elif slope_media < 0 or slope_mediana < 0:
    print("\n⚠️ Solo una frecuencia muestra descenso significativo → posible fatiga parcial.")
else:
    print("\n📈 No hay tendencia descendente clara → no se observa fatiga muscular evidente.")

```
evidnciando la siguiente grafica:

<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/82d8345b-dc18-4022-a99c-cee562449624" />

🔹 Pendiente de la Frecuencia Media: -0.0106

🔹 Pendiente de la Frecuencia Mediana: 0.0563

Las variaciones en las frecuencias media y mediana del EMG constituyen un indicador sensible de fatiga muscular. Una disminución progresiva en estos parámetros suele asociarse con una menor velocidad de conducción en las fibras musculares, así como con un cambio en el patrón de reclutamiento hacia unidades motoras de contracción más lenta. En el conjunto de datos analizado, los cambios observados son moderados, lo que sugiere que el músculo conserva un rendimiento funcional estable, con apenas signos incipientes de fatiga fisiológica.
Cabe señalar que esta estabilidad también podría deberse a una limitación en la medición, como el uso de un módulo orientado al registro del latido cardíaco en lugar de las contracciones musculares, lo cual afectaría la sensibilidad del análisis electromiográfico.


## Parte C 

![Diagramas lab 4_page-0003](https://github.com/user-attachments/assets/61307227-83f8-4225-9303-267050daeab5)

Para esta ultima parte se aplica la transformada rápida de Fourier (FFT) a cada contracción de la señal y se realiza la gráfica de el espectro de amplitud comparando las primeras contracciones con la ultimas utilizando el siguiente código:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq

# =====================================================
# 1️⃣ CONFIGURACIÓN Y CARGA DE DATOS
# =====================================================
ruta_txt = "/senal_EMG_captura_2.txt"  # <-- cambia si es necesario
fs = 1000                              # Frecuencia de muestreo [Hz]

# Cargar la señal
data = np.loadtxt(ruta_txt)
t = data[:, 0]
x = data[:, 1]

# =====================================================
# 2️⃣ FILTRADO PASA BANDA (20–450 Hz)
# =====================================================
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def aplicar_filtro(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, data)

x_filt = aplicar_filtro(x, 20, 450, fs)

# =====================================================
# 3️⃣ SEGMENTACIÓN AUTOMÁTICA DE CONTRACCIONES
# =====================================================
x_rect = np.abs(x_filt - np.mean(x_filt))
b, a = butter(2, 2/(fs/2), btype='low')
env = filtfilt(b, a, x_rect)
env_norm = env / np.max(env)

umbral = np.mean(env_norm) + 1.2*np.std(env_norm)
activa = env_norm > umbral

start_idx = np.where(np.diff(activa.astype(int)) == 1)[0]
end_idx   = np.where(np.diff(activa.astype(int)) == -1)[0]

if len(end_idx) > 0 and end_idx[0] < start_idx[0]:
    end_idx = end_idx[1:]
if len(start_idx) > len(end_idx):
    start_idx = start_idx[:-1]

min_len = int(0.15 * fs)
contracciones = [(i, f) for i, f in zip(start_idx, end_idx) if (f - i) > min_len]

print(f"🔹 Se detectaron {len(contracciones)} contracciones")

# =====================================================
# 4️⃣ FFT POR CONTRACCIÓN
# =====================================================
def calcular_fft(signal, fs):
    N = len(signal)
    freqs = fftfreq(N, 1/fs)
    fft_vals = np.abs(fft(signal)) / N
    mask = freqs > 0  # solo frecuencias positivas
    return freqs[mask], fft_vals[mask]

# =====================================================
# 5️⃣ COMPARAR ESPECTROS: PRIMERAS VS ÚLTIMAS CONTRACCIONES
# =====================================================
num_mostrar = 3  # número de contracciones iniciales/finales a comparar

primeras = contracciones[:num_mostrar]
ultimas  = contracciones[-num_mostrar:]

plt.figure(figsize=(12,6))

# ----- Primeras contracciones
for idx, (ini, fin) in enumerate(primeras, 1):
    f, mag = calcular_fft(x_filt[ini:fin], fs)
    plt.plot(f, mag, label=f"Inicio c{idx}", alpha=0.7)

# ----- Últimas contracciones
for idx, (ini, fin) in enumerate(ultimas, 1):
    f, mag = calcular_fft(x_filt[ini:fin], fs)
    plt.plot(f, mag, '--', label=f"Final c{len(contracciones)-num_mostrar+idx}", alpha=0.7)

plt.xlim(0, 250)  # rango típico EMG útil
plt.title("Comparación del espectro EMG - Primeras vs Últimas Contracciones")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud (a.u.)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =====================================================
#  ANÁLISIS DE FATIGA
# =====================================================
# Calcular frecuencia media para cada contracción
freqs_medias = []
for ini, fin in contracciones:
    f, mag = calcular_fft(x_filt[ini:fin], fs)
    f_media = np.sum(f * mag) / np.sum(mag)
    freqs_medias.append(f_media)

# Graficar tendencia de la frecuencia media
plt.figure(figsize=(8,4))
plt.plot(freqs_medias, 'o-', color='purple')
plt.title("Tendencia de la Frecuencia Media - Fatiga Muscular")
plt.xlabel("Número de Contracción")
plt.ylabel("Frecuencia Media (Hz)")
plt.grid(True)
plt.tight_layout()
plt.show()
```
Obteniendo las gráficas que se muestran a continuación:
<img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/029433cc-b34a-4d19-984d-bc8cef919d0c" />

<img width="788" height="390" alt="image" src="https://github.com/user-attachments/assets/dfd8b955-9e98-4b8a-b528-4252130b9cf3" />


En la primer gráfica podemos observar que las contracciones iniciales c1, c2 y c3 tienen una amplitud de magnitud entre 0,003 y 0,006 y las contracciones finales tienen una amplitud mayor siendo esta de una magnitud entre  0,006 Y 0,008 esto nos indica que en las contracciones iniciales como hay una menor actividad al estar el músculo en reposo entonces hay una menor magnitud en la gráfica, por otro lado en las contracciones finales como hay mayor activacion de unidades motoras aumenta la magnitud lo que demuestra la presencia de fatiga en el músculo.
Adicionalmente podemos observar en la segunda gráfica que cuando el músculo tiene una mayor intensidad en la contracción la frecuancia media es alta , pero evidenciamos además que la frecuencia disminuye cuando el músculo está fatigado debido a que el potencial de acción se propaga más lento generando menos picos de alta frecuencia.

El análisis espectral como herramienta diagnóstica en electromiografía es muy útil ya que el contenido de alta frecuencia y el desplazaiento de los picos hacia altas o bajas frecuencias permiten identificar la activación muscular, la fatiga como lo observamos en este laboratorio pero además permite detectar neuropatías o miopatías en el músculo, por ejemplo detectando actividad cuando el músculo está en reposo. 



