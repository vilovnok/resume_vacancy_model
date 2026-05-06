# HR Model


Загружаем данные для тестирования
```bash
curl -L -o data.zip "https://www.kaggle.com/api/v1/datasets/download/vilovnok/dataset-test"

unzip data.zip -d ./src/minio-seed
```


## Запуск сервиса

### 1. Клонирование репозитория
```bash
git clone https://github.com/user/repo.git
cd hr_matcher/app/
```

### 2. Заполнить .env
```

```

### 3. Поднять контейнеры
```
docker compose up -d
```

### 4. Протестировать ручки (optional)
```
http://localhost:8080/docs#/
```


## Эксперименты
Предварительно необходимо распаковать архив для создания директории `benchmarks/raw/hr/` с набором данных и внедрить в директорию `experiments` для проведения экспериментов.