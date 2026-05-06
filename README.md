# HR Matcher

Сервис для векторизации и поиска резюме / вакансий.

---

## 🚀 Быстрый старт

### 1. Клонирование репозитория

```bash
git clone https://github.com/user/repo.git
cd hr_matcher/src
```

2. Настройка окружения

```bash
# Создайте .env файл на основе шаблона:

cp service/.env.sample service/.env
cp airflow/.env.sample airflow/.env
```

## Подготовка данных и модели


1. Скачать датасет
```bash
curl -L -o data.zip "https://www.kaggle.com/api/v1/datasets/download/vilovnok/dataset-test"
```

2. Распаковать данные и распределить по соответствутющим директориям 
```
unzip data.zip -d ./src/minio-seed

mv $(find ./src/minio-seed -name "*.onnx") ./src/service/models/model.onnx

find ./src/minio-seed -name "*.onnx" -delete
```

---

Итоговая структура
```
src/
├── service/
│   └── models/
│       └── model.onnx
│
├── minio-seed/
│   ├── resumes.zip
│   ├── vacancies.zip
│   ├── *.csv
```

---


## Запуск системы
```bash
docker compose up -d
```

## 2. Проверка статуса
```bash
docker ps
```

## Доступ к сервисам

| Сервис        | URL                                                      |
| ------------- | -------------------------------------------------------- |
| FastAPI       | [http://localhost:8080/docs](http://localhost:8080/docs) |
| Airflow       | [http://localhost:9080](http://localhost:9080)           |
| MinIO Console | [http://localhost:9001](http://localhost:9001)           |
| Prometheus    | [http://localhost:9090](http://localhost:9090)           |
| Grafana       | [http://localhost:3000](http://localhost:3000)           |


## Эксперименты
Предварительно необходимо распаковать архив для создания директории `benchmarks/raw/hr/` с набором данных и внедрить в директорию `experiments` для проведения экспериментов.