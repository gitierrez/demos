import logging
import optimum

from optimum.onnxruntime import ORTQuantizer, ORTModelForSequenceClassification
from optimum.onnxruntime.configuration import AutoQuantizationConfig


def quantize_model(
    model: optimum.onnxruntime.ORTModel,
    save_model_to: str,
    quantization_config: str | optimum.onnxruntime.QuantizationConfig = 'default',
    model_class: type[optimum.onnxruntime.ORTModel] = ORTModelForSequenceClassification
):
    quantizer = ORTQuantizer.from_pretrained(model)
    if quantization_config == 'default':
        dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    else:
        dqconfig = quantization_config

    quantizer.quantize(save_dir=save_model_to, quantization_config=dqconfig)
    logging.info(f'Quantized model saved to {save_model_to}')
    return model_class.from_pretrained(save_model_to, from_transformers=True)
