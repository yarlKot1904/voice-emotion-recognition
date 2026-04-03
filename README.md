# Voice emotion recognition (Dusha)

Пайплайн SER на корпусе [Dusha Crowd](https://github.com/salute-developers/golos/tree/master/dusha#dusha-dataset): 4 класса (`angry`, `sad`, `neutral`, `positive`).

## Подготовка

1. Python 3.10+, виртуальное окружение.
2. `pip install -r requirements.txt` и `pip install -e .` из корня репозитория (чтобы работали импорты `src.*`).
3. Скачать и распаковать `crowd.tar`, задать корень датасета:

```bash
set DUSHA_ROOT=C:\path\to\dusha_root
```

Структура должна содержать `crowd_train/raw_crowd_train.jsonl`, `crowd_train/wavs/`, `crowd_test/...`.

## Манифест и сплиты

```bash
python -m src.data.build_manifest --dataset-root %DUSHA_ROOT% --output-dir data/processed
```

Опции отладки: `--max-total 5000`, `--max-per-class 1000`.

## Обучение

```bash
python -m src.training.train --model cnn --epochs 30 --tensorboard
```

MLP baseline: `--model mlp`.

Чекпоинт: `checkpoints/ser/best.pt` (или `--exp-name`, `--checkpoint-dir`).

### Улучшение качества (рекомендуется)

- Больше данных: пересоберите манифест **без** `--max-total` (или с большим лимитом).
- По умолчанию для CNN включены **аугментации** (сдвиг по времени + шум); отключить: `--no-augment`.
- Более ёмкая модель: `--cnn-variant large` (по умолчанию уже `large`).
- Пример «усиленного» прогона:

```bash
python -m src.training.train --model cnn --cnn-variant large --class-weights --scheduler --label-smoothing 0.05 --tensorboard
```

Дополнительно: `--weight-decay 1e-4`, `--grad-clip 1.0` (по умолчанию), снижение LR при плато: `--scheduler`. Старые чекпоинты без поля `cnn_variant` в `evaluate`/`predict` считаются архитектурой `base`.

### Ускорение обучения

Узкое место — **CPU**: `librosa` в `Dataset` на каждый шаг. На **CUDA** по умолчанию включён **AMP** (см. сообщение при старте); отключить: `--no-amp`.

**Mel на GPU (рекомендуется при CUDA):** флаг **`--mel-on-gpu`** — в батч попадает только сырой wav, log-mel считается **torchaudio** на GPU внутри модели. Обычно заметно быстрее полного librosa-пайплайна. Чекпоинт помечается `feature_config.mel_backend: torchaudio`; `evaluate` / `predict` подхватывают сами. Старые веса, обученные с librosa, с этим режимом не совместимы.

```bash
python -m src.training.train --model cnn --mel-on-gpu --cnn-variant large --batch-size 64 ...
```

- **`--num-workers`**: на **Windows** по умолчанию **0** (мультипроцессный DataLoader часто тормозит/зависает). На Linux можно **4–8**.
- **`--batch-size 64`** (или выше, если хватает VRAM) — меньше итераций за эпоху.
- **Короче аудио / реже кадры STFT** (меньше работы на сэмпл):
  `--max-length-sec 2 --hop-length 768` или `--hop-length 1024`
  (чуть грубее по времени, но быстрее; в чекпоинт пишется `feature_config` для `evaluate`/`predict`).
- **PyTorch 2+**: `--compile` может ускорить шаг на GPU после прогрева (экспериментально).

Пример быстрого прогона:

```bash
python -m src.training.train --model cnn --cnn-variant large --batch-size 64 --max-length-sec 2 --hop-length 768 --class-weights --scheduler --label-smoothing 0.05 --tensorboard --exp-name ser_fast
```

На Linux при необходимости добавьте `--num-workers 4`. На Windows оставьте значение по умолчанию (`0`), иначе возможны зависания.

## Оценка

```bash
python -m src.training.evaluate --checkpoint checkpoints/ser/best.pt
```

## Инференс

```bash
python -m src.inference.predict --checkpoint checkpoints/ser/best.pt --wav path\to\file.wav
```

## Демо (Streamlit)

```bash
streamlit run demo/app.py
```

## Podcast (опционально)

Сырые wav для Podcast не публикуются; доступны предрасчитанные `.npy` в `features.tar`. См. [src/data/podcast_features.py](src/data/podcast_features.py) и `FEATURES_ROOT` в `src/config.py`.

## Лицензия данных

См. [лицензию Dusha](https://github.com/salute-developers/golos/tree/master/dusha#license).
