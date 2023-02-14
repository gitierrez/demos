import datasets
import optimum
import transformers
from optimum.pipelines import pipeline as onnx_pipeline
from transformers import pipeline as transformers_pipeline

from .benchmark import PerformanceBenchmark
from .onnx_benchmark import ONNXPerformanceBenchmark


def calculate_statistics(
    models: dict[str, [transformers.PreTrainedModel | optimum.onnxruntime.ORTModel]],
    dataset: datasets.Dataset,
    metrics: list[str] = None,
    text_column: str = 'text',
    label_column: str = 'labels',
):
    results = {}
    for name, model in models.items():
        if isinstance(model, optimum.onnxruntime.ORTModel):
            pipe = onnx_pipeline('text-classification', model=model, accelerator='ort')
            benchmark = ONNXPerformanceBenchmark
        else:
            pipe = transformers_pipeline('text-classification', model=model)
            benchmark = PerformanceBenchmark
        results[name] = {
            'latency': benchmark.compute_latency(pipe),
            'metrics': benchmark.compute_metrics(pipe, dataset, metrics, text_column, label_column),
            'size': benchmark.compute_size(pipe),
        }
    return results
