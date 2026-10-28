# Projeto 7 - respostas computacionais dos exercícios 1 a 6 (gera os gráficos para a lista)
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz, iirpeak, iirnotch, sawtooth
from scipy.io import wavfile
from suaBibSignal import signalMeu

fs = 44100
sinal = signalMeu()


def filtro_IIR_ordem2(x, b, a):
    """Implementação manual (como no PDF): y[n] = b0 x[n] + b1 x[n-1] + b2 x[n-2] - a1 y[n-1] - a2 y[n-2]"""
    xp = np.concatenate(([0, 0], x))      # amostras anteriores ao início = 0
    yp = np.zeros(len(x) + 2)
    for n in range(2, len(xp)):
        yp[n] = (b[0] * xp[n] + b[1] * xp[n - 1] + b[2] * xp[n - 2]
                 - a[1] * yp[n - 1] - a[2] * yp[n - 2])
    y = yp[2:]
    return y


def ganho_em(xf, yf, f):
    i = np.argmin(np.abs(xf - f))
    return yf[i]


def ex1():
    print("\n=== Exercício 1: passa-baixa 500 Hz ===")
    b, a = butter(N=2, Wn=500 / (fs / 2), btype='low')
    print("b =", b, "\na =", a)

    t = np.arange(0, 3, 1 / fs)
    x = 0.5 * sawtooth(2 * np.pi * 200 * t)         # 200 Hz + harmônicos (400, 600, 800 ...)
    y = filtro_IIR_ordem2(x, b, a)
    print("diferença máx. para scipy.lfilter:", np.max(np.abs(y - lfilter(b, a, x))))

    xf, X = sinal.calcFFT(x, fs)
    _, Y = sinal.calcFFT(y, fs)
    print("Harmônico   original    filtrado   variação")
    for f in [200, 400, 600, 1000, 2000, 3000, 5000]:
        o, fl = ganho_em(xf, X, f), ganho_em(xf, Y, f)
        print("{:6d} Hz  {:9.5f}  {:9.5f}  {:7.1f} dB".format(f, o, fl, 20 * np.log10(fl / o)))

    fig, ax = plt.subplots(2, 2, figsize=(11, 6))
    ax[0, 0].plot(t[:441] * 1000, x[:441]); ax[0, 0].set_title("Original (tempo)")
    ax[0, 1].plot(t[:441] * 1000, y[:441]); ax[0, 1].set_title("Filtrado (tempo)")
    for a_, Z, tt in [(ax[1, 0], X, "FFT original"), (ax[1, 1], Y, "FFT filtrado")]:
        a_.plot(xf, Z); a_.set_xlim(0, 5000); a_.set_ylim(0, 0.25)
        a_.axvline(500, color='r', linestyle=':', label='500 Hz'); a_.legend(); a_.set_title(tt)
        a_.set_xlabel("Frequência (Hz)")
    for a_ in ax[0]:
        a_.set_xlabel("Tempo (ms)")
    fig.tight_layout(); fig.savefig("ex1_fft.png", dpi=120)

    wavfile.write("ex1_original.wav", fs, (x * 32767).astype(np.int16))
    wavfile.write("ex1_filtrado.wav", fs, (y / np.max(np.abs(y)) * 0.5 * 32767).astype(np.int16))
    try:
        import sounddevice as sd
        sd.play(x, fs); sd.wait(); sd.play(y / np.max(np.abs(y)) * 0.5, fs); sd.wait()
    except Exception:
        print("(sounddevice indisponível: ouça ex1_original.wav e ex1_filtrado.wav)")


def ex2_3():
    print("\n=== Exercícios 2 e 3 ===")
    print("Limiar da audição:", 10 * np.log10(1e-12 / 1e-12), "dB")
    print("Limiar da dor    :", 10 * np.log10(1 / 1e-12), "dB")
    print("Ex.3 ganho:", 20 * np.log10(10 / 20), "dB")


def ex4():
    print("\n=== Exercício 4 ===")
    b, a = butter(2, 500 / (fs / 2))
    def ganho(w_rad):
        _, h = freqz(b, a, worN=[w_rad / fs])   # worN em rad/amostra
        return abs(h[0])
    for w in (1e4, 1e3):
        g = ganho(w)
        print("w = {:.0e} rad/s: ganho {:.1f} dB ({:.3f}) -> saída {:.2f}".format(
            w, 20 * np.log10(g), g, 50 * g))
    ws = np.logspace(2.5, 4.3, 20000)
    gs = np.array([ganho(w) for w in ws])
    i6 = np.argmin(np.abs(20 * np.log10(gs) + 6))
    i3 = np.argmin(np.abs(20 * np.log10(gs) + 3))
    print("-6 dB em w = {:.0f} rad/s ({:.0f} Hz), razão Vs/Ve = {:.3f}".format(ws[i6], ws[i6] / 2 / np.pi, gs[i6]))
    print("-3 dB em w = {:.0f} rad/s ({:.0f} Hz), razão Vs/Ve = {:.3f}".format(ws[i3], ws[i3] / 2 / np.pi, gs[i3]))


def largura_3dB(w, h_db, pico=True):
    ref = np.max(h_db) if pico else 0
    alvo = ref - 3 if pico else -3
    idx = np.where(h_db >= alvo)[0] if pico else np.where(h_db <= alvo)[0]
    return w[idx[0]], w[idx[-1]]


def ex5_6():
    print("\n=== Exercício 5 (passa-faixa, iirpeak) ===")
    f0 = 2000
    plt.figure(figsize=(8, 4))
    for Q in (1, 10, 50):
        b, a = iirpeak(w0=f0 / (fs / 2), Q=Q)
        w, h = freqz(b, a, worN=40000, fs=fs)
        hdb = 20 * np.log10(np.abs(h) + 1e-12)
        lo, hi = largura_3dB(w, hdb)
        print("Q={:3d}: banda de -3 dB de {:.0f} a {:.0f} Hz (largura {:.0f} Hz ~ f0/Q = {:.0f})".format(
            Q, lo, hi, hi - lo, f0 / Q))
        plt.semilogx(w[1:], hdb[1:], label="Q = {}".format(Q))
    plt.axvline(f0, color='r', linestyle=':'); plt.ylim(-60, 5); plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.title("Passa-faixa 2 kHz - efeito de Q"); plt.xlabel("Frequência (Hz)"); plt.ylabel("Magnitude (dB)")
    plt.legend(); plt.tight_layout(); plt.savefig("ex5_bode_Q.png", dpi=120)

    print("\n=== Exercício 6 (rejeita-faixa, iirnotch) ===")
    plt.figure(figsize=(8, 4))
    for f0n, Q in [(2000, 1), (2000, 10), (2000, 50), (1000, 10)]:
        b, a = iirnotch(w0=f0n / (fs / 2), Q=Q)
        w, h = freqz(b, a, worN=40000, fs=fs)
        hdb = 20 * np.log10(np.abs(h) + 1e-12)
        lo, hi = largura_3dB(w, hdb, pico=False)
        print("f0={:4d} Hz, Q={:3d}: banda rejeitada (-3 dB) {:.0f} a {:.0f} Hz (largura {:.0f} Hz)".format(
            f0n, Q, lo, hi, hi - lo))
        plt.semilogx(w[1:], np.maximum(hdb[1:], -60), label="f0 = {} Hz, Q = {}".format(f0n, Q))
    plt.ylim(-60, 5); plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.title("Rejeita-faixa - efeito de Q e da frequência central"); plt.xlabel("Frequência (Hz)")
    plt.ylabel("Magnitude (dB)"); plt.legend(); plt.tight_layout(); plt.savefig("ex6_bode_notch.png", dpi=120)


if __name__ == "__main__":
    ex1(); ex2_3(); ex4(); ex5_6()
    plt.show()
