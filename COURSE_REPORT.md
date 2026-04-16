# Курсовая работа

## Тема

**Разработка системы распознавания эмоций по голосу на основе глубокого обучения**

Дисциплина: машинное обучение / искусственный интеллект  
Студент: `<ФИО студента>`  
Группа: `<номер группы>`  
Руководитель: `<ФИО руководителя>`  
Год: 2026

---

## Аннотация

В курсовой работе разработан программный прототип системы Speech Emotion Recognition (SER), предназначенный для определения эмоционального состояния говорящего по аудиозаписи речи. В качестве основного набора данных используется Dusha Crowd, содержащий русскоязычные аудиофайлы с разметкой по четырём классам эмоций: `angry`, `sad`, `neutral`, `positive`.

В ходе работы реализован полный воспроизводимый ML-пайплайн: подготовка CSV-манифеста, загрузка и нормализация аудио, извлечение log-mel признаков, обучение свёрточных нейронных сетей, оценка качества и демонстрационное Streamlit-приложение для загрузки WAV-файла и вывода вероятностей классов.

Лучший сохранённый чекпоинт `checkpoints/ser_resnet_gpu_5ep_v2/best.pt` на тестовой выборке показал:

- Accuracy: **0.7502**
- Macro F1: **0.6575**
- Weighted F1: **0.74**

---

## Содержание

1. Постановка задачи
2. Используемый набор данных
3. Архитектура программной системы
4. Предобработка аудио и признаки
5. Модели машинного обучения
6. Обучение и воспроизводимость
7. Экспериментальные результаты
8. Демонстрационное приложение
9. Инструкция по запуску
10. Выводы и направления развития
11. Приложение: команды и артефакты

---

## 1. Постановка задачи

Цель работы: разработать систему классификации эмоций по аудиосигналу речи.

Вход системы: WAV-аудиозапись или файл, содержащий речь человека.  
Выход системы: предсказанная эмоциональная метка и распределение вероятностей по классам:

- `angry`
- `sad`
- `neutral`
- `positive`

Для достижения цели были поставлены следующие задачи:

1. Подготовить датасет и сформировать воспроизводимые train/validation/test split-файлы.
2. Реализовать загрузку, нормализацию и аугментацию аудиосигнала.
3. Реализовать извлечение признаков MFCC и log-mel spectrogram.
4. Реализовать нейросетевые модели для классификации эмоций.
5. Провести обучение и оценку моделей по accuracy, macro F1 и confusion matrix.
6. Реализовать CLI-инференс и простое веб-демо.

---

## 2. Используемый набор данных

В работе используется датасет **Dusha Crowd** из проекта Golos/Dusha:

- язык: русский;
- формат: WAV-аудиофайлы и JSONL-разметка;
- домен: Crowd;
- целевые классы: `angry`, `sad`, `neutral`, `positive`;
- ссылка: <https://github.com/salute-developers/golos/tree/master/dusha#dusha-dataset>.

После обработки были сформированы CSV-файлы:

- `data/processed/splits/train.csv`
- `data/processed/splits/val.csv`
- `data/processed/splits/test.csv`

Распределение классов:

| Split | Всего | angry | sad | neutral | positive |
|---|---:|---:|---:|---:|---:|
| Train | 163900 | 18909 | 26719 | 101746 | 16526 |
| Validation | 18212 | 2101 | 2969 | 11306 | 1836 |
| Test | 16953 | 1991 | 2912 | 10343 | 1707 |

Набор данных заметно несбалансирован: класс `neutral` занимает основную долю выборки. Поэтому в работе дополнительно анализируется macro F1, который лучше отражает качество по миноритарным классам.

---

## 3. Архитектура программной системы

Проект организован модульно:

| Модуль | Назначение |
|---|---|
| `src/data/build_manifest.py` | построение общего манифеста и train/val/test split-файлов |
| `src/data/dataset.py` | PyTorch Dataset для признаков и waveform-режима |
| `src/data/preprocessing.py` | загрузка WAV, crop/pad, аугментации |
| `src/features/audio_features.py` | MFCC и log-mel признаки через librosa |
| `src/models/mlp_model.py` | MLP baseline на MFCC |
| `src/models/cnn_model.py` | CNN-модели: `base`, `large`, `resnet` |
| `src/models/mel_frontend.py` | torchaudio log-mel frontend внутри модели |
| `src/training/train.py` | обучение, checkpointing, TensorBoard |
| `src/training/evaluate.py` | оценка модели на split-файле |
| `src/inference/predict.py` | CLI-инференс одного WAV-файла |
| `demo/app.py` | Streamlit-веб-интерфейс |

Основной сценарий работы:

1. JSONL-разметка Dusha преобразуется в CSV-манифест.
2. Dataset загружает WAV, приводит его к моно и фиксированной длительности.
3. Для CNN используется log-mel spectrogram.
4. Модель выдаёт logits по четырём классам.
5. Для инференса logits преобразуются в вероятности через softmax.

---

## 4. Предобработка аудио и признаки

Основные параметры аудио:

| Параметр | Значение |
|---|---:|
| Sample rate | 16000 Hz |
| FFT window | 1024 |
| Hop length в лучшем checkpoint | 640 |
| Mel bands | 64 |
| Длительность фрагмента в лучшем checkpoint | 2.5 сек |

Для аудио реализованы следующие операции:

- загрузка WAV в mono;
- resampling до 16 kHz;
- crop или padding до фиксированной длины;
- случайный crop во время обучения;
- waveform-аугментации: временной сдвиг, изменение громкости, добавление слабого шума;
- log-mel нормализация по sample.

В проекте поддержаны два режима вычисления log-mel:

- `librosa` в Dataset;
- `torchaudio` внутри модели при флаге `--mel-on-gpu`.

Для GPU-обучения использовался режим `--mel-on-gpu`, так как он уменьшает нагрузку на CPU и позволяет считать спектрограмму внутри PyTorch pipeline.

---

## 5. Модели машинного обучения

В проекте реализованы несколько архитектур.

### 5.1. MLP baseline

MLP используется как простая базовая модель на flattened MFCC-признаках. Она полезна для проверки работоспособности пайплайна, но не является основной моделью для финального качества.

### 5.2. CNN `base`

Базовая свёрточная модель для log-mel spectrogram. Содержит несколько Conv2D + BatchNorm + ReLU блоков и классификационную голову.

### 5.3. CNN `large`

Увеличенная версия CNN с большим числом каналов. Использовалась как промежуточный вариант для улучшения качества.

### 5.4. CNN `resnet`

Финальная версия модели. Она использует residual-блоки, BatchNorm, SiLU-активации, dropout и pooling по статистикам. Входом является log-mel spectrogram размерности `[B, 1, n_mels, T]`.

Преимущества `resnet`-варианта:

- устойчивее обучается на log-mel спектрограммах;
- сохраняет локальные спектрально-временные признаки;
- residual-связи улучшают прохождение градиента;
- компактная классификационная голова снижает риск переобучения.

---

## 6. Обучение и воспроизводимость

Основной тренировочный скрипт: `src/training/train.py`.

Используемые механизмы:

- фиксированный seed через `RANDOM_SEED`;
- сохранение лучшего checkpoint по validation accuracy;
- сохранение `feature_config` и `training_config` в checkpoint;
- поддержка TensorBoard;
- поддержка `Adam` и `AdamW`;
- поддержка `OneCycleLR`, `ReduceLROnPlateau`, `CosineAnnealingLR`;
- gradient accumulation через `--grad-accum-steps`;
- автоматический выбор устройства через `src/runtime/device.py`.

На рабочей машине использовалась видеокарта **NVIDIA GeForce GTX 1050 Ti** с compute capability `sm_61`. Для данной карты AMP/FP16 отключается автоматически, так как FP16-режим приводил к `nan` loss. Обучение и оценка выполняются на GPU в FP32.

Финальная команда обучения лучшего checkpoint:

```powershell
.\.venv\Scripts\python.exe -m src.training.train `
  --model cnn `
  --epochs 5 `
  --cnn-variant resnet `
  --mel-on-gpu `
  --batch-size 4 `
  --grad-accum-steps 4 `
  --max-length-sec 2.5 `
  --hop-length 640 `
  --lr 1e-3 `
  --lr-schedule onecycle `
  --exp-name ser_resnet_gpu_5ep_v2
```

Параметры лучшего checkpoint:

| Параметр | Значение |
|---|---|
| Checkpoint | `checkpoints/ser_resnet_gpu_5ep_v2/best.pt` |
| Model | CNN `resnet` |
| Backend признаков | `torchaudio` |
| Sample rate | 16000 |
| Max length | 2.5 сек |
| Hop length | 640 |
| n_mels | 64 |
| Optimizer | AdamW |
| LR schedule | OneCycleLR |
| Batch size | 4 |
| Gradient accumulation | 4 |
| Effective batch size | 16 |
| Label smoothing | 0.0 |
| SpecAugment | off |

---

## 7. Экспериментальные результаты

Оценка выполнялась командой:

```powershell
.\.venv\Scripts\python.exe -m src.training.evaluate `
  --checkpoint checkpoints\ser_resnet_gpu_5ep_v2\best.pt `
  --batch-size 16 `
  --num-workers 0
```

Также была выполнена оценка на validation split и сравнение со старым checkpoint `ser_1050ti_quality`.

Логи оценок сохранены в:

- `artifacts/reports/ser_resnet_gpu_5ep_v2_test.txt`
- `artifacts/reports/ser_resnet_gpu_5ep_v2_val.txt`
- `artifacts/reports/ser_1050ti_quality_test.txt`

### 7.1. Сводная таблица

| Checkpoint | Split | Accuracy | Macro F1 | Комментарий |
|---|---|---:|---:|---|
| `ser_resnet_gpu_5ep_v2` | Validation | 0.7542 | 0.6591 | Финальная модель |
| `ser_resnet_gpu_5ep_v2` | Test | 0.7502 | 0.6575 | Финальная тестовая оценка |
| `ser_1050ti_quality` | Test | 0.6070 | 0.5305 | Старый checkpoint для сравнения |

### 7.2. Classification report для test split

Финальная модель `ser_resnet_gpu_5ep_v2`:

| Класс | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| angry | 0.65 | 0.61 | 0.63 | 1991 |
| sad | 0.72 | 0.53 | 0.61 | 2912 |
| neutral | 0.79 | 0.88 | 0.83 | 10343 |
| positive | 0.64 | 0.50 | 0.56 | 1707 |
| Accuracy |  |  | 0.75 | 16953 |
| Macro avg | 0.70 | 0.63 | 0.66 | 16953 |
| Weighted avg | 0.74 | 0.75 | 0.74 | 16953 |

### 7.3. Confusion matrix для test split

Порядок классов: `angry`, `sad`, `neutral`, `positive`.

| True \ Pred | angry | sad | neutral | positive |
|---|---:|---:|---:|---:|
| angry | 1217 | 29 | 590 | 155 |
| sad | 57 | 1542 | 1278 | 35 |
| neutral | 410 | 538 | 9104 | 291 |
| positive | 196 | 42 | 614 | 855 |

### 7.4. Анализ результатов

Модель лучше всего распознаёт класс `neutral`, что ожидаемо из-за сильного преобладания этого класса в обучающей выборке. Классы `sad` и `positive` чаще всего путаются с `neutral`. Это типичная проблема для SER-задач: нейтральная речь часто спектрально близка к слабовыраженным эмоциональным состояниям, а сами классы в датасете представлены несбалансированно.

Класс `angry` распознаётся сравнительно устойчиво: recall составляет 0.61. Класс `positive` остаётся самым сложным среди целевых эмоций: recall равен 0.50, что указывает на необходимость дальнейшего улучшения данных или обучения.

Потенциальные направления улучшения:

- обучение больше 5 эпох;
- подбор class weights или focal loss для миноритарных классов;
- использование balanced sampler;
- более сильная модель CRNN или wav2vec-like encoder;
- подбор оптимальной длительности аудиофрагмента;
- анализ ошибок по тексту, полу, длительности и источнику разметки.

---

## 8. Демонстрационное приложение

Для демонстрации реализовано Streamlit-приложение `demo/app.py`.

Функции интерфейса:

1. ввод пути к checkpoint;
2. загрузка WAV-файла;
3. запуск инференса;
4. вывод предсказанного класса;
5. отображение вероятностей по всем четырём эмоциям.

Запуск:

```powershell
.\.venv\Scripts\python.exe -m streamlit run demo\app.py
```

В поле checkpoint рекомендуется указать:

```text
checkpoints\ser_resnet_gpu_5ep_v2\best.pt
```

### Места для скриншотов веб-интерфейса

> **Скриншот 1. Запуск Streamlit в терминале**  
> Вставить изображение: `docs/screenshots/01_streamlit_terminal.png`

![Скриншот 1. Запуск Streamlit в терминале](docs/screenshots/01_streamlit_terminal.png)

> **Скриншот 2. Главная страница веб-интерфейса**  
> Вставить изображение: `docs/screenshots/02_demo_main_page.png`

![Скриншот 2. Главная страница веб-интерфейса](docs/screenshots/02_demo_main_page.png)

> **Скриншот 3. Загрузка WAV-файла**  
> Вставить изображение: `docs/screenshots/03_demo_wav_upload.png`

![Скриншот 3. Загрузка WAV-файла](docs/screenshots/03_demo_wav_upload.png)

> **Скриншот 4. Результат распознавания эмоции**  
> Вставить изображение: `docs/screenshots/04_demo_prediction_result.png`

![Скриншот 4. Результат распознавания эмоции](docs/screenshots/04_demo_prediction_result.png)

> **Скриншот 5. TensorBoard с графиками обучения, если запускался `--tensorboard`**  
> Вставить изображение: `docs/screenshots/05_tensorboard_curves.png`

![Скриншот 5. TensorBoard с графиками обучения](docs/screenshots/05_tensorboard_curves.png)

---

## 9. Инструкция по запуску

### 9.1. Подготовка окружения

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

Для GTX 1050 Ti использовалась CUDA-сборка PyTorch `cu126`:

```text
torch 2.11.0+cu126
torchaudio 2.11.0+cu126
torchvision 0.26.0+cu126
```

### 9.2. Оценка модели

```powershell
.\.venv\Scripts\python.exe -m src.training.evaluate `
  --checkpoint checkpoints\ser_resnet_gpu_5ep_v2\best.pt `
  --batch-size 16 `
  --num-workers 0
```

### 9.3. Инференс одного файла

```powershell
.\.venv\Scripts\python.exe -m src.inference.predict `
  --checkpoint checkpoints\ser_resnet_gpu_5ep_v2\best.pt `
  --wav path\to\file.wav
```

Пример выбора первого файла из test split:

```powershell
$wav = (Import-Csv data\processed\splits\test.csv | Select-Object -First 1).path
.\.venv\Scripts\python.exe -m src.inference.predict `
  --checkpoint checkpoints\ser_resnet_gpu_5ep_v2\best.pt `
  --wav $wav
```

### 9.4. Запуск веб-демо

```powershell
.\.venv\Scripts\python.exe -m streamlit run demo\app.py
```

### 9.5. TensorBoard

```powershell
.\.venv\Scripts\python.exe -m tensorboard.main --logdir checkpoints
```

---

## 10. Выводы

В результате курсовой работы был создан рабочий прототип системы распознавания эмоций по голосу. Реализованы все ключевые этапы ML-пайплайна: подготовка данных, извлечение признаков, обучение моделей, оценка качества и демонстрация результата через веб-интерфейс.

Финальная модель на базе CNN `resnet` и log-mel spectrogram достигла accuracy **0.7502** и macro F1 **0.6575** на тестовой выборке. Данный результат показывает, что модель способна выделять эмоциональные признаки из русскоязычной речи, но качество по миноритарным и акустически близким классам требует дальнейшей оптимизации.

Главное ограничение текущего решения связано с дисбалансом классов: преобладание `neutral` улучшает общую accuracy, но снижает качество распознавания `sad` и `positive`. Для дальнейшего развития целесообразно использовать balanced sampling, дополнительные аугментации, более долгую тренировку и архитектуры, учитывающие временную динамику речи.

---

## 11. Приложение: проверенные команды и артефакты

### 11.1. Проверка Python-модулей

```powershell
.\.venv\Scripts\python.exe -m compileall -q src
```

### 11.2. Оценка финального checkpoint на test

```powershell
.\.venv\Scripts\python.exe -m src.training.evaluate `
  --checkpoint checkpoints\ser_resnet_gpu_5ep_v2\best.pt `
  --batch-size 16 `
  --num-workers 0
```

Результат сохранён в:

```text
artifacts/reports/ser_resnet_gpu_5ep_v2_test.txt
```

### 11.3. Оценка финального checkpoint на validation

```powershell
.\.venv\Scripts\python.exe -m src.training.evaluate `
  --checkpoint checkpoints\ser_resnet_gpu_5ep_v2\best.pt `
  --split data\processed\splits\val.csv `
  --batch-size 16 `
  --num-workers 0
```

Результат сохранён в:

```text
artifacts/reports/ser_resnet_gpu_5ep_v2_val.txt
```

### 11.4. Сравнительная оценка старого checkpoint

```powershell
.\.venv\Scripts\python.exe -m src.training.evaluate `
  --checkpoint checkpoints\ser_1050ti_quality\best.pt `
  --batch-size 16 `
  --num-workers 0
```

Результат сохранён в:

```text
artifacts/reports/ser_1050ti_quality_test.txt
```

### 11.5. Версии ключевых библиотек

| Библиотека | Версия |
|---|---|
| Python | `.venv`, проверено через локальный интерпретатор |
| torch | 2.11.0+cu126 |
| torchaudio | 2.11.0+cu126 |
| librosa | 0.11.0 |
| scikit-learn | 1.8.0 |
| streamlit | 1.56.0 |

