# Model Export and Quantization

Директория содержит скрипты для экспорта моделей в формат ONNX и их квантизации в INT8 для оптимизированного инференса.

## Export to ONNX
```
python deploy/export/export_onnx.py --model-path deploy/models/название_модели --output-dir service/model/название_модели
```

## Quantize to INT8
```
python deploy/export/quantize_int8.py service/model/model_f32.onnx --output service/model/model_int8.onnx
```
