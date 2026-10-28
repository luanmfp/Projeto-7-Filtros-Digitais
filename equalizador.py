# Projeto 7 - Equalizador de áudio de 12 bandas (filtros peaking EQ em cascata)
# Usa a função peaking_eq do notebook fornecido (filtro_peaking_EQ.ipynb).
#
#   python equalizador.py                 -> usa audio_teste.wav (se não existir, gera um sweep 20 Hz-20 kHz)
#   python equalizador.py meu_audio.wav   -> usa o seu áudio (.wav)
#
# Dependências: numpy, scipy, matplotlib, sounddevice  (tkinter já vem com o Python do Windows/macOS)

import os
import sys
import threading
import numpy as np
from scipy.signal import freqz, lfilter, chirp
from scipy.io import wavfile

BANDAS = [20, 32, 64, 125, 250, 500, 1000, 2000, 4000, 8000, 16000, 20000]   # Hz (figura do enunciado)
NOMES = ["20", "32", "64", "125", "250", "500", "1k", "2k", "4k", "8k", "16k", "20k"]
Q_BANDA = 1.41          # ~ 1 oitava de largura por banda (Q = f0 / largura de banda)
GANHO_MAX = 10          # dB (de -10 a +10)


# ---------------------------------------------------------------- filtro fornecido
def peaking_eq(f0, gain_db, Q, fs):
    """Filtro peaking EQ (biquad). Retorna os coeficientes (b, a) de G(z)."""
    A = 10 ** (gain_db / 40)
    omega = 2 * np.pi * f0 / fs
    alpha = np.sin(omega) / (2 * Q)

    b0 = 1 + alpha * A
    b1 = -2 * np.cos(omega)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(omega)
    a2 = 1 - alpha / A

    b = np.array([b0, b1, b2]) / a0
    a = np.array([a0, a1, a2]) / a0
    return b, a


# ---------------------------------------------------------------- núcleo do equalizador
def projetar_filtros(ganhos_db, fs, Q=Q_BANDA):
    """Um filtro peaking por banda. Bandas em/above Nyquist (fs/2) são ignoradas."""
    filtros = []
    for f0, g in zip(BANDAS, ganhos_db):
        if f0 >= fs / 2:
            continue
        b, a = peaking_eq(f0, g, Q, fs)
        filtros.append((f0, b, a))
    return filtros


def aplicar_filtros(x, filtros):
    """Aplica os filtros um a um (cascata): a saída de um é a entrada do próximo."""
    y = np.asarray(x, dtype=float)
    for _, b, a in filtros:
        y = lfilter(b, a, y, axis=0)
    pico = np.max(np.abs(y)) if y.size else 0
    if pico > 1:                       # evita saturar (clipping) na placa de som
        y = y / pico * 0.99
    return y


def tf_total(filtros):
    """G_total(z) = G1(z)*G2(z)*...*G12(z): multiplicar as funções de transferência
    equivale a multiplicar numeradores entre si e denominadores entre si (convolução)."""
    b_tot, a_tot = np.array([1.0]), np.array([1.0])
    for _, b, a in filtros:
        b_tot = np.convolve(b_tot, b)
        a_tot = np.convolve(a_tot, a)
    return b_tot, a_tot


def resposta_total(filtros, fs, n=2000):
    """Bode do filtro completo: |G_total| = produto dos |G_i| (em dB, soma dos dB).
    Calculado pelo produto das respostas individuais, que é numericamente mais estável
    que avaliar o polinômio de ordem 24 de tf_total()."""
    w = np.logspace(np.log10(10), np.log10(fs / 2 * 0.999), n)
    H = np.ones(n, dtype=complex)
    for _, b, a in filtros:
        _, h = freqz(b, a, worN=w, fs=fs)
        H *= h
    return w, 20 * np.log10(np.abs(H) + 1e-12)


# ---------------------------------------------------------------- áudio
def carregar_audio(caminho):
    fs, x = wavfile.read(caminho)
    if x.dtype == np.int16:
        x = x / 32768.0
    elif x.dtype == np.int32:
        x = x / 2147483648.0
    elif x.dtype == np.uint8:
        x = (x.astype(float) - 128) / 128.0
    else:
        x = x.astype(float)
    return fs, x


def gerar_audio_teste(caminho="audio_teste.wav", fs=44100, dur=8):
    """Sweep logarítmico de 20 Hz a 20 kHz: ideal para ouvir cada banda do equalizador."""
    t = np.linspace(0, dur, int(fs * dur), endpoint=False)
    x = 0.5 * chirp(t, f0=20, f1=min(20000, fs / 2 * 0.95), t1=dur, method="logarithmic")
    fade = int(0.05 * fs)
    x[:fade] *= np.linspace(0, 1, fade)
    x[-fade:] *= np.linspace(1, 0, fade)
    wavfile.write(caminho, fs, (x * 32767).astype(np.int16))
    return fs, x


# ---------------------------------------------------------------- interface
def iniciar_gui(caminho_audio):
    import tkinter as tk
    from tkinter import filedialog, messagebox
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import sounddevice as sd

    if os.path.exists(caminho_audio):
        fs, x = carregar_audio(caminho_audio)
    else:
        print("Áudio '{}' não encontrado: gerando sweep de teste.".format(caminho_audio))
        fs, x = gerar_audio_teste(caminho_audio)

    estado = {"fs": fs, "x": x, "parar": False}

    root = tk.Tk()
    root.title("Equalizador de áudio - 12 bandas")

    # --- sliders
    frame = tk.Frame(root)
    frame.pack(padx=10, pady=10)
    sliders, rotulos = [], []
    for i, nome in enumerate(NOMES):
        col = tk.Frame(frame)
        col.grid(row=0, column=i, padx=4)
        tk.Label(col, text=nome + " Hz" if i == 0 else nome).pack()
        s = tk.Scale(col, from_=GANHO_MAX, to=-GANHO_MAX, resolution=0.5, length=220,
                     orient=tk.VERTICAL, showvalue=False,
                     command=lambda _v: atualizar())
        s.set(0)
        s.pack()
        r = tk.Label(col, text="0.0 dB", width=7)
        r.pack()
        sliders.append(s)
        rotulos.append(r)

    # --- gráfico de Bode do filtro completo
    fig = Figure(figsize=(8, 3.2))
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    status = tk.Label(root, text="Áudio: {} ({} Hz, {:.1f} s)".format(
        os.path.basename(caminho_audio), fs, len(x) / fs))
    status.pack()

    def ganhos():
        return [s.get() for s in sliders]

    def filtros_atuais():
        return projetar_filtros(ganhos(), estado["fs"])

    def atualizar():
        for s, r in zip(sliders, rotulos):
            r.config(text="{:+.1f} dB".format(s.get()))
        w, g = resposta_total(filtros_atuais(), estado["fs"])
        ax.clear()
        ax.semilogx(w, g)
        ax.set_xlim(20, 20000)
        ax.set_ylim(-15, 15)
        ax.set_xlabel("Frequência (Hz)")
        ax.set_ylabel("Ganho (dB)")
        ax.set_title("Diagrama de Bode - filtro completo (12 bandas)")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        for f in BANDAS:
            ax.axvline(f, color="gray", linewidth=0.3)
        fig.tight_layout()
        canvas.draw_idle()

    def resetar():
        for s in sliders:
            s.set(0)
        atualizar()

    def tocar():
        parar()
        estado["parar"] = False
        filtros = filtros_atuais()
        y = aplicar_filtros(estado["x"], filtros)
        x0, fs0 = estado["x"], estado["fs"]

        def run():
            estado["msg"] = "Tocando ORIGINAL..."
            sd.play(x0, fs0)
            sd.wait()
            if not estado["parar"]:
                estado["msg"] = "Tocando FILTRADO..."
                sd.play(y, fs0)
                sd.wait()
            estado["msg"] = "Pronto."

        threading.Thread(target=run, daemon=True).start()

    def parar():
        estado["parar"] = True
        sd.stop()

    def abrir():
        c = filedialog.askopenfilename(filetypes=[("WAV", "*.wav")])
        if not c:
            return
        try:
            estado["fs"], estado["x"] = carregar_audio(c)
        except Exception as e:
            messagebox.showerror("Erro", str(e))
            return
        status.config(text="Áudio: {} ({} Hz)".format(os.path.basename(c), estado["fs"]))
        atualizar()

    def salvar():
        y = aplicar_filtros(estado["x"], filtros_atuais())
        wavfile.write("audio_filtrado.wav", estado["fs"], (y * 32767).astype(np.int16))
        status.config(text="Salvo em audio_filtrado.wav")

    def imprimir_tf():
        b, a = tf_total(filtros_atuais())
        np.set_printoptions(precision=6, linewidth=120)
        print("G_total(z) = B(z)/A(z), ordem {}".format(len(a) - 1))
        print("B(z) (numerador, potências decrescentes de z):\n", b)
        print("A(z) (denominador):\n", a)

    def atualizar_status():
        # a thread de áudio só escreve em estado["msg"]; a interface é atualizada aqui (thread principal)
        if estado.get("msg"):
            status.config(text=estado["msg"])
            estado["msg"] = None
        root.after(200, atualizar_status)

    botoes = tk.Frame(root)
    botoes.pack(pady=6)
    for texto, cmd in [("▶ Tocar original → filtrado", tocar), ("■ Parar", parar),
                       ("Resetar", resetar), ("Abrir .wav", abrir),
                       ("Salvar filtrado", salvar), ("Imprimir G(z) total", imprimir_tf)]:
        tk.Button(botoes, text=texto, command=cmd).pack(side=tk.LEFT, padx=4)

    atualizar()
    atualizar_status()
    root.mainloop()


if __name__ == "__main__":
    iniciar_gui(sys.argv[1] if len(sys.argv) > 1 else "audio_teste.wav")
