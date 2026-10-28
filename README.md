# Projeto 7 – Equalizador Digital de Áudio

Projeto da disciplina **Camada Física da Computação** (Engenharia da Computação – Insper).
Equalizador de áudio de **12 bandas** (−10 a +10 dB por banda), construído com filtros digitais *peaking EQ* em cascata, com interface gráfica, reprodução do áudio original e filtrado e diagrama de Bode do filtro completo.

## Funcionalidades

- 12 bandas nas frequências: 20, 32, 64, 125, 250, 500 Hz, 1, 2, 4, 8, 16 e 20 kHz.
- Cada banda é ajustada por um slider de −10 a +10 dB.
- Os filtros são aplicados ao sinal **um a um**, em cascata.
- O botão de tocar reproduz o áudio **original** e, em seguida, o **filtrado**, para comparar.
- O diagrama de Bode do **filtro completo** é atualizado em tempo real ao mexer nos sliders.
- Também é possível abrir outro `.wav`, salvar o áudio filtrado (`audio_filtrado.wav`) e imprimir no terminal a função de transferência total G(z).

## Estrutura

| Arquivo | Descrição |
|---|---|
| `equalizador.py` | Equalizador completo (filtros + interface + Bode) |
| `exercicios_filtros.py` | Código dos exercícios 1 a 6 da lista; gera os gráficos `ex*.png` |
| `suaBibSignal.py` | Biblioteca de apoio (geração de senoides e FFT), usada pelos exercícios |
| `requirements.txt` | Dependências Python |
| `ex1_fft.png`, `ex5_bode_Q.png`, `ex6_bode_notch.png` | Gráficos gerados pelos exercícios |

## Instalação

```bash
pip install -r requirements.txt
```

O `tkinter` (interface) já vem com o Python no Windows e no macOS. No Linux: `sudo apt install python3-tk`.

## Como usar

```bash
python equalizador.py                  # usa audio_teste.wav (se não existir, gera um sweep de 20 Hz a 20 kHz)
python equalizador.py meu_audio.wav    # usa um áudio seu (formato .wav)
```

1. Ajuste os sliders de cada banda.
2. Clique em **▶ Tocar original → filtrado** para ouvir o áudio original e depois o equalizado.
3. Observe o Bode do filtro completo no gráfico.

Para os exercícios da lista:

```bash
python exercicios_filtros.py
```

## Como funciona

### Filtro de cada banda

Cada banda usa o filtro *peaking EQ* (biquad) do notebook fornecido (`filtro_peaking_EQ.ipynb`), com função de transferência discreta de 2ª ordem:

```
G(z) = (b0 + b1·z⁻¹ + b2·z⁻²) / (1 + a1·z⁻¹ + a2·z⁻²)
```

Os coeficientes dependem da frequência central `f0`, do ganho em dB, do fator de qualidade `Q` e da taxa de amostragem `fs`. O projeto usa **Q = 1,41**, o que dá bandas de cerca de uma oitava.

Bandas em ou acima da frequência de Nyquist (`fs/2`) são ignoradas automaticamente.

### Aplicação no sinal

Os 12 filtros são aplicados em sequência (`scipy.signal.lfilter`): a saída de um é a entrada do próximo. Se o resultado ultrapassar a amplitude 1, ele é normalizado para evitar saturação (*clipping*) na placa de som.

### Filtro completo (função de transferência equivalente)

Filtros em cascata têm função de transferência igual ao produto das individuais:

```
G_total(z) = G1(z) · G2(z) · … · G12(z)
```

Isso equivale a convolver os numeradores entre si e os denominadores entre si (`tf_total`), resultando em um filtro de **ordem 24**. Em dB, o ganho total é a **soma** dos ganhos em dB de cada banda.

O diagrama de Bode é calculado pelo **produto das respostas em frequência individuais**, e não avaliando o polinômio de ordem 24. Os dois são matematicamente equivalentes, mas o polinômio de ordem alta é numericamente mal condicionado (principalmente com bandas de baixa frequência), e a avaliação direta produz erros visíveis.

## Observações

- A banda de 20 kHz fica próxima do limite de Nyquist quando `fs = 44100 Hz`, então o comportamento dela é mais limitado.
- Bandas vizinhas se sobrepõem; por isso, ao aumentar várias bandas seguidas, o ganho resultante pode passar de +10 dB nessa região (o Bode mostra isso).
- Os arquivos `.wav` suportados são PCM de 8, 16 ou 32 bits e float, mono ou estéreo.
