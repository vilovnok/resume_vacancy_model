## Эксперименты

Перед запуском экспериментов необходимо:

1. Подготовить датасет в директории `benchmarks/raw/hr/`
2. Установить зависимости из `experiments/requirements.txt`
3. Настроить `accelerate`

```bash
pip install -r experiments/requirements.txt

accelerate config
```

---

## Структура экспериментов

```text
experiments/
├── train_stages.sh
├── test_stages.sh
├── stage1/
├── stage3/
├── stage4/
└── stage5/
```

---

## Exp. 1–2 — Bi-Encoder Retrieval

Эксперименты с базовой bi-encoder архитектурой:

- baseline без дообучения
- supervised fine-tuning retrieval модели

### Обучение

```bash
bash train_stages.sh stage1
```

### Тестирование

```bash
bash test_stages.sh stage1
```

---

## Exp. 3 — Multi-View Representation Learning

Дообучение retrieval-модели с использованием multi-view представлений и дополнительных позитивных/негативных пар.

### Обучение

```bash
bash train_stages.sh stage3
```

### Тестирование

```bash
bash test_stages.sh stage3
```

---

## Exp. 4 — Knowledge Distillation via Embeddings

Дистилляция знаний через согласование векторных представлений teacher/student моделей.

### Обучение

```bash
bash train_stages.sh stage4
```

### Тестирование

```bash
bash test_stages.sh stage4
```

---

## Exp. 5 — Hidden-State Knowledge Distillation

Дистилляция знаний через внутренние скрытые представления модели.

### Обучение

```bash
bash train_stages.sh stage5
```

### Тестирование

```bash
bash test_stages.sh stage5
```

---

## Запуск всех экспериментов

### Полный цикл обучения

```bash
bash train_stages.sh all
```

### Полный цикл тестирования

```bash
bash test_stages.sh all
```