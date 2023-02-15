import datasets

from optimum.pipelines import pipeline as onnx_pipeline
from transformers import pipeline as transformers_pipeline

from .benchmark import PerformanceBenchmark
from .onnx_benchmark import ONNXPerformanceBenchmark


def calculate_statistics(
    models: dict[str, str],
    dataset: datasets.Dataset,
    metrics: list[str] = None,
    text_column: str = 'text',
    label_column: str = 'labels',
):
    results = {}
    for name, model in models.items():
        try:
            pipe = transformers_pipeline('text-classification', model=model)
            benchmark = PerformanceBenchmark
        except ValueError:
            pipe = onnx_pipeline('text-classification', model=model, accelerator='ort')
            benchmark = ONNXPerformanceBenchmark

        results[name] = {
            'latency': benchmark.compute_latency(pipe),
            'metrics': benchmark.compute_metrics(
                pipeline=pipe,
                dataset=dataset,
                metrics=metrics,
                text_column=text_column,
                label_column=label_column
            ),
            'size': benchmark.compute_size(pipe),
        }
    return results
