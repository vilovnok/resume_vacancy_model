# HR Matcher

Сервис для векторизации и поиска резюме / вакансий.

---

## Init

### 1. Клонирование репозитория

```bash
git clone https://github.com/user/repo.git
cd hr_matcher/src
```

2. Настройка окружения

```bash
# Создайте .env файл на основе шаблона:

cp src/service/.env.sample src/service/.env
cp src/airflow/.env.sample src/airflow/.env
```

## Подготовка данных и модели


1. Скачать датасет
```bash
curl -L -o data.zip "https://www.kaggle.com/api/v1/datasets/download/vilovnok/dataset-test"
```

2. Распаковать данные и распределить по соответствутющим директориям 
```bash
unzip data.zip -d ./src/minio-seed

mv $(find ./src/minio-seed -name "*.onnx") ./src/service/models/model.onnx

find ./src/minio-seed -name "*.onnx" -delete
```

---

Итоговая структура
```
hr_matcher/
│
├── README.md
├── benchmarks/                
├── experiments/              
│
├── src/
│
│   ├── ./                    
│   │   ├── docker-compose.yml
│   │   ├── otel-collector-config.yml
│   │   ├── prometheus.yml
│   │
│   ├── airflow/             
│   │   ├── dags/
│   │   ├── tasks/
│   │   ├── scripts/
│   │   ├── config.py
│   │   ├── requirements.txt
│   │   ├── Dockerfile
│   │   └── webserver_config.py
│   │
│   ├── service/             
│   │   ├── main.py          
│   │   ├── engine/          
│   │   ├── models/          
│   │   ├── worker.py        
│   │   ├── cache.py
│   │   ├── metrics.py
│   │   ├── utils.py
│   │   ├── config.py
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   │
│   └── minio-seed/               
│       └── raw/ & processed/  
│    
│
└── README.md
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

Скачать датасет
```bash
curl -L -o benchmarks.zip "https://www.kaggle.com/api/v1/datasets/download/vilovnok/benchmarks"

unzip benchmarks.zip -d ./experiments/benchmarks/
```

