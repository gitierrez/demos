from .benchmark import PerformanceBenchmark


class ONNXPerformanceBenchmark(PerformanceBenchmark):

    @staticmethod
    def _compute_size(model, tmp_path):
        model.save_pretrained(tmp_path)
        model_path = tmp_path.joinpath('model.onnx')
        return model_path.stat().st_size / (1024 * 1024)
