import torch
import datasets
import numpy as np
import warnings

from pathlib import Path
from time import perf_counter


class PerformanceBenchmark:

    @classmethod
    def compute_metrics(
        cls,
        pipeline,
        dataset,
        metrics: list[str],
        text_column: str = 'text',
        label_column: str = 'labels'
    ):
        preds, labels = [], []
        class_labels = dataset.features[label_column]
        for i, sample in enumerate(dataset):
            predicted_label = pipeline(sample[text_column])[0][label_column]
            predicted_id = class_labels.str2int(predicted_label)
            label_id = sample[label_column]
            preds.append(predicted_id)
            labels.append(label_id)

        out_metrics = {}
        for metric in metrics:
            metric = datasets.load_metric(metric)
            out_metrics[metric.name] = metric.compute(predictions=preds, references=labels)
        return out_metrics

    @classmethod
    def compute_size(cls, pipeline) -> float:
        tmp_path = Path('model')
        size = cls._compute_size(pipeline.model, tmp_path)
        try:
            tmp_path.unlink()
        except PermissionError:  # common issue if running in Windows
            warnings.warn(
                f'Captured PermissionError when attempting to remove {tmp_path}'
            )
        return size

    @staticmethod
    def _compute_size(model, tmp_path):
        model_path = tmp_path.joinpath('model.pt')
        torch.save(model.state_dict(), model_path)
        return tmp_path.stat().st_size / (1024 * 1024)

    @classmethod
    def compute_latency(cls, pipeline, query: str = 'auto', num_runs: int = 100, unit: str = 'ms'):
        if unit != 'ms':
            raise NotImplementedError('Only ms supported for now')
        if query == 'auto':
            query = 'what is the process for making a vacation request'
        cls._warmup_run(pipeline, query)
        latencies = []
        for _ in range(num_runs):
            start = perf_counter()
            _ = pipeline(query)
            end = perf_counter()
            latency = end - start
            latencies.append(latency)
        return {
            'mean': 1000 * np.mean(latencies),
            'std': 1000 * np.std(latencies),
            'min': 1000 * np.min(latencies),
            'max': 1000 * np.max(latencies)
        }

    @staticmethod
    def _warmup_run(pipeline, query: str, steps: int = 10):
        for _ in range(steps):
            _ = pipeline(query)
