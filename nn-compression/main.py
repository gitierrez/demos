# knowledge distillation
# quantization
# graph optimization
import gc
import json
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import datasets
import transformers
import optimum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from time import perf_counter

from optimum.pipelines import pipeline
from optimum.onnxruntime import ORTQuantizer, ORTModelForSequenceClassification
from optimum.onnxruntime.configuration import AutoQuantizationConfig


DEBUG = False


class PerformanceBenchmark:

    def __init__(self, pipeline: transformers.pipeline, dataset: datasets.Dataset, metric: str = 'accuracy'):
        self.pipeline = pipeline
        self.dataset = dataset
        self.metric = datasets.load_metric(metric)
        self.class_labels = self.dataset.features['intent']

    def compute_statistics(self):
        stats = self.compute_accuracy()
        stats['size'] = self.compute_size()
        mean_time, std_time = self.compute_latency()
        stats['latency'] = mean_time
        stats['latency_std'] = std_time
        stats['metadata'] = {
            'metric': self.metric.__class__.__name__,
        }
        return stats

    def compute_accuracy(self):
        preds, labels = [], []
        for i, sample in enumerate(self.dataset):
            predicted_label = self.pipeline(sample['text'])[0]['label']
            predicted_id = self.class_labels.str2int(predicted_label)
            label_id = sample['intent']
            preds.append(predicted_id)
            labels.append(label_id)
        metric = self.metric.compute(predictions=preds, references=labels)
        return metric

    def compute_size(self):
        if isinstance(self.pipeline.model, optimum.onnxruntime.ORTModel):
            tmp_path = Path('tmp-onnx')
            self.pipeline.model.save_pretrained(tmp_path)
            model_path = tmp_path.joinpath('model.onnx')
            size_in_mb = model_path.stat().st_size / (1024 * 1024)
        else:
            tmp_path = Path('model.pt')
            state_dict = self.pipeline.model.state_dict()
            torch.save(state_dict, tmp_path)
            size_in_mb = tmp_path.stat().st_size / (1024 * 1024)
        try:
            tmp_path.unlink()
        except PermissionError:
            warnings.warn(
                f'Captured PermissionError when attempting to remove {tmp_path}'
            )
        return size_in_mb

    def compute_latency(self, query: str = 'auto', num_runs: int = 100):
        if query == 'auto':
            query = 'what is the process for making a vacation request'
        self._warmup_run(query)
        latencies = []
        for _ in range(num_runs):
            start = perf_counter()
            _ = self.pipeline(query)
            end = perf_counter()
            latency = end - start
            latencies.append(latency)
        mean_time_in_ms = 1000 * np.mean(latencies)
        std_time_in_ms = 1000 * np.std(latencies)
        return mean_time_in_ms, std_time_in_ms

    def _warmup_run(self, query: str, steps: int = 10):
        for _ in range(steps):
            _ = self.pipeline(query)


class KLDistillationTrainingArguments(transformers.TrainingArguments):

    def __init__(self, *args, alpha: float = 0.5, temperature: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature


class KLDistillationTrainer(transformers.Trainer):

    def __init__(self, *args, teacher: transformers.PreTrainedModel, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher

    def compute_loss(self, model, inputs, return_outputs=False):
        student_outputs = model(**inputs)
        loss_ce = student_outputs.loss
        student_logits = student_outputs.logits
        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs)
            teacher_logits = teacher_outputs.logits
        loss_kld = (self.args.temperature ** 2) * nn.functional.kl_div(
            input=F.log_softmax(student_logits / self.args.temperature, dim=-1),
            target=F.softmax(teacher_logits / self.args.temperature, dim=-1),
            reduction='batchmean'
        )
        loss = self.args.alpha * loss_ce + (1.0 - self.args.alpha) * loss_kld
        return (loss, student_outputs) if return_outputs else loss


base_ckpt = 'transformersbook/bert-base-uncased-finetuned-clinc'
student_ckpt = 'distilbert-base-uncased'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pipe = transformers.pipeline('text-classification', model=base_ckpt, device=device)
clinc = datasets.load_dataset('clinc_oos', 'plus')

intents_from_base = clinc['test'].features['intent']
num_classes_from_base = intents_from_base.num_classes
id2label_from_base = pipe.model.config.id2label
label2id_from_base = pipe.model.config.label2id

if DEBUG:
    for split in clinc:
        clinc[split] = clinc[split].select(range(100))


# distillation
student_tokenizer = transformers.AutoTokenizer.from_pretrained(student_ckpt)


def tokenize_text(batch):
    return student_tokenizer(batch['text'], truncation=True)


clinc_enc = clinc.map(tokenize_text, batched=True, remove_columns=['text'])
clinc_enc = clinc_enc.rename_column('intent', 'labels')


def compute_metrics(pred):
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=1)
    return datasets.load_metric('accuracy').compute(predictions=predictions, references=labels)


save_student_to = 'distilbert-base-uncased-finetuned-clinc'
student_training_args = KLDistillationTrainingArguments(
    output_dir=save_student_to,
    evaluation_strategy='epoch',
    num_train_epochs=10,
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    alpha=0.125,
    temperature=7,
    weight_decay=0.01,
)

# these also work
intents_from_enc = clinc_enc['test'].features['labels']
num_classes_from_enc = intents_from_enc.num_classes

student_config = transformers.AutoConfig.from_pretrained(
    student_ckpt,
    num_labels=num_classes_from_base,
    id2label=id2label_from_base,
    label2id=label2id_from_base,
)

# distillation training
def student_model_init():
    return transformers.AutoModelForSequenceClassification.from_pretrained(
        student_ckpt,
        config=student_config
    ).to(device)


trainer = KLDistillationTrainer(
    model_init=student_model_init,
    teacher=pipe.model,
    args=student_training_args,
    train_dataset=clinc_enc['train'],
    eval_dataset=clinc_enc['validation'],
    compute_metrics=compute_metrics,
    tokenizer=student_tokenizer,
)
trainer.train()
trainer.save_model(save_student_to)


# this raises OOM
"""def hp_space(trial):
    gc.collect()
    return {
        'num_train_epochs': trial.suggest_int('num_train_epochs', 5, 10),
        'alpha': trial.suggest_float('alpha', 0, 1),
        'temperature': trial.suggest_int('temperature', 2, 20),
    }

n_trials = 20
if DEBUG:
    n_trials = 2

best_run = trainer.hyperparameter_search(
    n_trials=n_trials,
    direction='maximize',
    hp_space=hp_space,
)

for k, v in best_run.hyperparameters.items():
    setattr(student_training_args, k, v)

optim_trainer = KLDistillationTrainer(
    model_init=student_model_init,
    teacher=pipe.model,
    args=student_training_args,
    train_dataset=clinc_enc['train'],
    eval_dataset=clinc_enc['validation'],
    compute_metrics=compute_metrics,
    tokenizer=student_tokenizer,
)

optim_trainer.train()
optim_trainer.save_model(save_student_to)"""


pipe = transformers.pipeline('text-classification', model=base_ckpt)

benchmark = {}
# default
benchmark['bert-base'] = PerformanceBenchmark(pipeline=pipe, dataset=clinc['test']).compute_statistics()

# distillation
distilled_pipe = transformers.pipeline('text-classification', model=save_student_to)
benchmark['distilbert'] = PerformanceBenchmark(pipeline=distilled_pipe, dataset=clinc['test']).compute_statistics()

# distillation + onnx
ort_model = ORTModelForSequenceClassification.from_pretrained(save_student_to, from_transformers=True)

onnx_pipe = pipeline('text-classification', model=ort_model, accelerator='ort')
benchmark['distilbert+onnx'] = PerformanceBenchmark(pipeline=onnx_pipe, dataset=clinc['test']).compute_statistics()

# distillation + onnx + quantization
quantizer = ORTQuantizer.from_pretrained(ort_model)
dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
quantizer.quantize(save_dir='onnx-dq', quantization_config=dqconfig)

dq_pipe = pipeline('text-classification', model='onnx-dq', accelerator='ort')
benchmark['distilbert+onnx+quant'] = PerformanceBenchmark(pipeline=dq_pipe, dataset=clinc['test']).compute_statistics()

json.dump(benchmark, open('export.json', 'w'))


def plot_benchmark(benchmark: dict):
    df = pd.DataFrame.from_dict(benchmark, orient='index')
    for idx in df.index:
        model_stats = df.loc[idx]
        plt.scatter(
            x=model_stats['latency'],
            y=model_stats['accuracy']*100,
            s=model_stats['size'],
            label=idx,
            alpha=0.5,
        )

    legend = plt.legend(bbox_to_anchor=(1, 1))
    for handle in legend.legendHandles:
        handle.set_sizes([20])
    plt.xlabel('Latency (ms)')
    plt.ylabel('Accuracy (%)')
    plt.show()


plot_benchmark(benchmark)
