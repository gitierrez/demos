import json
import datasets
import transformers

from optimum.onnxruntime import ORTModelForSequenceClassification

from compression import distill_model, quantize_model
from benchmark import calculate_statistics


# load intent detection dataset
dataset = datasets.load_dataset('clinc_oos', 'plus')

# load base BERT model
teacher_model = transformers.AutoModelForSequenceClassification.from_pretrained(
    'transformersbook/bert-base-uncased-finetuned-clinc'
)
student_ckpt = 'distilbert-base-uncased'  # use DistilBERT as student model

# knowledge distillation
distilled_model = distill_model(
    teacher_model=teacher_model,
    student_ckpt=student_ckpt,
    dataset=dataset,
    save_model_to='distilbert-base-uncased-finetuned-clinc',
    label_column='intent',
)

# graph optimization
optimized_model = ORTModelForSequenceClassification.from_pretrained(
    'distilbert-base-uncased-finetuned-clinc',
    from_transformers=True
)
optimized_model.save_pretrained('distilbert-base-uncased-finetuned-clinc-onnx')

# quantization
quantized_model = quantize_model(
    model=optimized_model,
    save_model_to='distilbert-base-uncased-finetuned-clinc-onnx-quantized',
)

benchmark_results = calculate_statistics(
    models={
        'bert-base': 'transformersbook/bert-base-uncased-finetuned-clinc',
        'distilbert': 'distilbert-base-uncased-finetuned-clinc',
        'distilbert+onnx': 'distilbert-base-uncased-finetuned-clinc-onnx',
        'distilbert+onnx+dq': 'distilbert-base-uncased-finetuned-clinc-onnx-quantized',
    },
    dataset=dataset['test'],
    metrics=['accuracy'],
    text_column='text',
    label_column='intent',
)

json.dump(benchmark_results, open('benchmark_results.json', 'w'), indent=4)
