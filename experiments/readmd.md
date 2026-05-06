## Эксперименты

### Exp. (1-2)

Эксперименты без дообучения и дообучением моделей с архитектурой bi-encoder.

```bashs
bash train_stage1.sh

bash test_stage1.sh
```


### Exp. 3

Дообучение модели на основе архитектуры multi-view representation learn-
ing. 

```bashs
bash train_stage3.sh

bash test_stage3.sh
```

### Exp. 4

Дистилляция знаний через векторные представления
модели.

```bashs
bash train_stage4.sh

bash test_stage4.sh
```

### Exp. 5
Дистилляция знаний осуществляемое через внутренние представления
модели.

```bashs
bash train_stage5.sh

bash test_stage5.sh
```