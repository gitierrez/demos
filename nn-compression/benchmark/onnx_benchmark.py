import logging
from pathlib import Path

from .benchmark import PerformanceBenchmark


class ONNXPerformanceBenchmark(PerformanceBenchmark):

    @classmethod
    def compute_size(cls, pipeline):
        tmp_path = Path('model')
        pipeline.model.save_pretrained(tmp_path)
        model_path = tmp_path.joinpath('model.onnx')
        size = model_path.stat().st_size / (1024 * 1024)
        try:
            tmp_path.unlink()
        except PermissionError:  # common issue if running in Windows
            logging.warning(
                f'Captured PermissionError when attempting to remove {model_path}'
            )
        return size
