# Analisi del flusso e dell'elaborazione dei dati nel modello WaveNet: rete neurale autoregressiva per la sintesi di segnali audio

**Tesi di laurea — Alberto Di Ronza, A.A. 2024/25**

## Abstract

La tesi analizza **WaveNet**, il modello generativo autoregressivo per la sintesi di segnali audio grezzi (raw audio) basato su convoluzioni causali dilatate. L'obiettivo non è solo descrivere l'architettura, ma capire *come* il modello elabora i dati lungo la rete — quantizzazione mu-law, blocchi residuali con skip connection, meccanismo di dilatazione che espande il receptive field — e valutarne criticamente le scelte progettuali attraverso una fase sperimentale: addestramento su un dataset musicale nuovo, test di inferenza sistematici variando temperatura/regolarizzazione/input, analisi spettrale degli output e un sondaggio di valutazione soggettiva con ascoltatori reali.

![Struttura del modello WaveNet](immagini%20tesi/struttura_wavenet.png)

*Struttura del modello WaveNet (Aäron van den Oord et al., WaveNet: A Generative Model for Raw Audio, 2016) — Figura 2.1 della tesi.*

## Basato su pytorch-wavenet

L'architettura del modello, il training loop e l'algoritmo di generazione veloce (fast generation tramite code dilatate) sono basati sull'implementazione open source **[pytorch-wavenet](https://github.com/vincentherrmann/pytorch-wavenet)** di Vincent Herrmann (licenza MIT). Da quella base provengono `wavenet_model.py`, `wavenet_modules.py`, `wavenet_training.py`, `audio_data.py`, `model_logging.py` e `optimizers.py`; il checkpoint pre-addestrato sulla Ciaccona di Bach (`snapshots/snapshot_violini/`, 2017) è quello fornito dall'implementazione di riferimento, usato come base per tutti i test di inferenza del capitolo 3.

Il contributo originale della tesi, sviluppato sopra questa base, riguarda:

- la **costruzione di un dataset ad hoc** di musica house/techno a partire dal Freesound Loop Dataset e il relativo esperimento di training (`json_manage.py`);
- la **pipeline di analisi** del segnale usata per le conclusioni discusse in tesi: spettrogrammi, forma d'onda e distribuzione di probabilità del campionamento (`analisi.py`, `analisi_stime_spec.py`);
- gli esperimenti sistematici di inferenza (variazione di temperatura, regolarizzazione, tipo di input) e il sondaggio di valutazione soggettiva su 19 ascoltatori;
- gli script di **visualizzazione del grafo computazionale** (`visualize.py`, `wavenet_structure.py`).

Il repository contiene inoltre `gen_hist_plot.py` e `hist_tutte_frequenze_gen.py`, che stimano la frequenza fondamentale con l'algoritmo **YIN** e ne costruiscono istogrammi aggregati su più generazioni: sono strumenti di analisi esplorativa presenti nel codice, ma i risultati che ne derivano **non sono discussi nelle conclusioni della tesi**, che si basa invece sull'ispezione degli spettrogrammi e sul sondaggio soggettivo.

## Architettura del modello

| Parametro | Valore |
|---|---|
| Layer per blocco | 12 |
| Numero di blocchi | 4 |
| Kernel size (convoluzione dilatata) | 2 |
| Canali dilatazione | 32 |
| Canali residual | 64 |
| Canali skip | 512 |
| Canali finali | 512 |
| Livelli di quantizzazione (classi mu-law) | 256 |
| Sample rate | 16 000 Hz |
| Receptive field | 16 381 campioni (~1.02 s) |
| Learning rate / weight decay | 0.0001 / 0.0 |

Il receptive field di ~1 secondo è determinato dalla struttura a convoluzioni causali dilatate: la dilatazione raddoppia a ogni layer all'interno di un blocco (1, 2, 4, ..., 2¹¹) e si resetta a ogni nuovo blocco, permettendo alla rete di condizionare ogni campione generato su una finestra di contesto ampia con un numero di parametri molto più contenuto rispetto a una convoluzione non dilatata equivalente.

![Convoluzioni dilatate nel modello WaveNet](immagini%20tesi/convoluzioni_dilatate_wavenet.png)

*Visualizzazione delle convoluzioni dilatate nel modello WaveNet (Aäron van den Oord et al., WaveNet: A Generative Model for Raw Audio, 2016) — Figura 2.2 della tesi: ogni layer raddoppia la dilatazione rispetto al precedente, fino al reset a inizio blocco.*

## Esperimento 1 — Training su un dataset house/techno (risultato negativo)

Per verificare se il modello riesce ad apprendere facilmente le feature di un genere musicale più complesso e dinamico di quello di riferimento (musica classica), è stato costruito un dataset ad hoc a partire dal **Freesound Loop Dataset**: dei 9 455 loop disponibili (~3 000 corredati da annotazioni JSON con tempo, tonalità e genere), sono stati filtrati quelli di genere house/techno con tempo tra 100 e 110 BPM, ottenendo **203 file audio per circa 53 minuti totali** (`json_manage.py`).

Con l'architettura standard (12 layer, 4 blocchi ⇒ receptive field ~1.0 s) e oltre **100 000 step / 10 epoche** di training, il modello **non ha raggiunto convergenza**: in generazione produce quasi solo rumore o segnali silenziosi. L'ipotesi discussa in tesi è che il receptive field, pur esteso dalle dilatazioni, resti insufficiente per un genere così denso ritmicamente: a 100–110 BPM una battuta in 4/4 dura circa 2.18 s, più del doppio della finestra di contesto disponibile — a cui si aggiungono i vincoli computazionali che hanno impedito di provare configurazioni più profonde o training più lunghi.

## Esperimento 2 — Inferenza sul modello pre-addestrato (Ciaccona di Bach in Re minore)

Per valutare il comportamento in fase di inferenza è stato usato il checkpoint pre-addestrato sulla Ciaccona di Bach, condizionando la rete su una finestra reale di 3085 campioni (~0.193 s, coerente con l'`item_length` di training) e generando 60 000 campioni di output (~3.75 s). Sono stati fatti variare sistematicamente:

- **temperatura** ∈ {0.0, 0.2, 0.5, 0.8, 1.0, 1.2}, con 10 generazioni per valore, a innesco fissato (sinusoide a 440 Hz);
- **fattore di regolarizzazione**, testato a temperatura 0.0 e 1.0;
- **tipo di input**: segnale costante a zero, sinusoide, rumore gaussiano (σ = 0.25), finestre reali estratte dal dataset.

![Spettrogramma dell'audio di innesco a 440 Hz](immagini%20tesi/input_spectrogram_440hz.png) ![Spettrogramma dell'audio generato a partire dall'innesco a 440 Hz](immagini%20tesi/generated_spectrogram_440hz.png)

*Sinistra: spettrogramma dell'innesco sinusoidale a 440 Hz (La4) usato come seed. Destra: spettrogramma dei ~4 s generati autoregressivamente a partire da quell'innesco.*

**Effetto della temperatura.** A temperatura 0.0 la rete è completamente deterministica (sceglie sempre il livello a probabilità massima) e produce forme d'onda identiche a parità di innesco. Aumentando la temperatura la generazione diventa progressivamente più stocastica, fino a raggiungere a 1.2 una saturazione spettrale pressoché uniforme, simile a rumore bianco, con perdita della struttura temporale.

![Distribuzione di probabilità di un campione generato](immagini%20tesi/prob_distribution_sample.png)

*Distribuzione di probabilità sui 256 livelli di quantizzazione per un campione generato: a temperature basse la probabilità si concentra su pochi livelli adiacenti.*

**Effetto della regolarizzazione.** Il modello è risultato estremamente sensibile a questo parametro: per qualunque valore diverso da zero (tranne un'unica combinazione) il regolarizzatore abbassa i logit fino a far collassare la distribuzione sul livello di quantizzazione centrale (127), producendo segnali quasi completamente silenziosi. L'unica combinazione che ha prodotto output udibile è stata temperatura 1.0 con regolarizzazione 0.01 — segnale ascoltabile ma molto ripetitivo e rumoroso, privo di struttura armonica.

**Effetto del tipo di input.** Anche con inneschi banali (silenzio, sinusoide, rumore) il modello genera output coerenti con il timbro del dataset di addestramento (violino), a riprova che l'informazione appresa durante il training pesa più della natura dell'innesco stesso.

**Bias sistematico in frequenza.** Indipendentemente da temperatura e input, gli spettrogrammi generati mostrano una attenuazione sistematica dell'energia nella banda 50–250 Hz e nell'intorno degli 8 kHz, con energia concentrata nelle bande 500 Hz–1 kHz e 2–4 kHz — pattern coerente con il contenuto spettrale del dataset di addestramento, quindi verosimilmente un bias appreso piuttosto che un artefatto della generazione.

## Valutazione soggettiva (sondaggio, 19 ascoltatori)

Per completare l'analisi spettrale, un campione di 19 persone ha valutato qualità e rumorosità di segnali generati a diverse temperature:

| Temperatura | Ottima | Buona | Cattiva | Pessima |
|---|---|---|---|---|
| 0.0 | 21.1% | 73.7% | 0% | 5.3% |
| 0.5 | 0% | 21.1% | 78.9% | 0% |
| 0.8 | 0% | 52.6% | 47.4% | 0% |
| 1.0 | 0% | 36.8% | 47.4% | 15.8% |
| 1.2 | 0% | 10.5% | 0% | 89.5% |

Il risultato non è monotono: la temperatura 0.5 è quella percepita peggio (78.9% "Cattiva"), peggio persino di 1.0. La **temperatura 0.8** emerge come miglior compromesso tra naturalezza e stocasticità (52.6% "Buona", e il 68.4% dei rispondenti l'ha preferita rispetto a 0.5 in un confronto diretto). A 1.2 la qualità crolla quasi del tutto (89.5% "Pessima").

Infine, in un test di discriminazione tra una traccia generata e una estratta realmente dal dataset, il **68.4%** degli ascoltatori ha riconosciuto correttamente quella generata da WaveNet: il modello riproduce in modo riconoscibile il timbro del violino, ma i segnali restano percettivamente distinguibili da quelli reali nella maggioranza dei casi.

## Conclusioni principali

- Le convoluzioni dilatate estendono il receptive field esponenzialmente con la profondità senza aumentare il costo computazionale, ma il valore raggiunto (~1.0 s con 12 layer × 4 blocchi) può comunque risultare insufficiente per generi musicali ritmicamente densi (house/techno a 100–110 BPM): il training su questo dataset non converge.
- Sul modello pre-addestrato sulla Ciaccona di Bach, la temperatura è il parametro che più influenza qualità percepita e casualità della generazione, con un massimo di gradimento non monotono attorno a 0.8 (non a 0.0, nonostante quest'ultima produca l'output "più pulito" perché deterministico).
- Il coefficiente di regolarizzazione è un parametro molto sensibile: fuori da una stretta finestra di valori il modello collassa su output quasi silenziosi.
- Il modello apprende un bias sistematico in frequenza legato al dataset di training, indipendente dai parametri di generazione usati in inferenza.
- Il 68.4% degli ascoltatori distingue correttamente audio generato da audio reale: buona qualità percepita, ma non ancora indistinguibile dal dato reale.

## Struttura del repository

```
pytorch-wavenet-tesi/
├── wavenet_model.py         # architettura del modello (base: pytorch-wavenet)
├── wavenet_modules.py       # code dilatate e moduli per la fast generation (base: pytorch-wavenet)
├── wavenet_training.py      # training loop e validazione (base: pytorch-wavenet)
├── audio_data.py            # pre-processing e Dataset PyTorch (base: pytorch-wavenet)
├── model_logging.py         # logging TensorBoard (base: pytorch-wavenet)
├── optimizers.py            # ottimizzatori (base: pytorch-wavenet)
├── train_script.py          # configurazione ed esecuzione del training
├── generate_script.py       # configurazione ed esecuzione della generazione
├── test_script.py           # script di test
├── analisi.py                    # analisi spettrale (spettrogrammi, forma d'onda, distribuzioni)
├── analisi_stime_spec.py         # analisi delle caratteristiche spettrali in fase di test finale
├── gen_hist_plot.py               # istogrammi delle frequenze generate (YIN, esplorativo)
├── hist_tutte_frequenze_gen.py    # istogrammi aggregati su tutte le generazioni (YIN, esplorativo)
├── json_manage.py           # filtraggio e costruzione del dataset house/techno da annotazioni JSON
├── visualize.py              # visualizzazione del grafo computazionale (con wavenet_structure.py)
├── wavenet_structure.py
├── train_samples/bach_chaconne/   # dataset .npz (Bach Chaconne + dataset house/techno)
├── toy/                      # clip audio grezze del dataset house/techno (Freesound Loop Dataset)
├── snapshots/                # checkpoint: snapshot_violini (pre-addestrato, riferimento) e snapshot_toy (training di questa tesi)
├── logs/chaconne_model/      # log TensorBoard, per esperimento
└── generati/                 # output di generazione (audio, grafici, spettrogrammi)

immagini tesi/                # immagini usate in questo README, tratte dalla tesi
```

## Requirements

- Python 3.12.11
- PyTorch 2.4.1
- NumPy 1.26
- librosa 0.10
- Jupyter 1.0.0
- TensorFlow 2.16.2 e TensorBoard 2.16.2 (per il logging)
- soundfile 0.12

```bash
conda create -n wavenet python=3.12.11
conda activate wavenet
conda install pytorch=2.4.1 torchvision torchaudio cpuonly -c pytorch
conda install numpy=1.26 jupyter=1.0 -c conda-forge
conda install -c conda-forge librosa=0.10 soundfile=0.12.1
pip install tensorflow tensorboard
```

Per monitorare il training con TensorBoard:

```bash
tensorboard --logdir=logs/chaconne_model/logs_toy --host localhost --port 8088
```

## Bibliografia essenziale

- A. van den Oord et al., *WaveNet: A Generative Model for Raw Audio*, DeepMind, 2016 — [deepmind.google/blog/wavenet-a-generative-model-for-raw-audio](https://deepmind.google/blog/wavenet-a-generative-model-for-raw-audio)
- Vincent Herrmann, [pytorch-wavenet](https://github.com/vincentherrmann/pytorch-wavenet) — implementazione PyTorch di riferimento (MIT license)
- Freesound Loop Dataset — dataset di loop musicali con annotazioni, usato per l'esperimento sul genere house/techno
