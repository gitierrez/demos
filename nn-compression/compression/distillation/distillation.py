import torch
import logging
import datasets
import transformers
import numpy as np

from .kullback_leibler import KnowledgeDistillationTrainingArguments, KnowledgeDistillationTrainer


class TextClassificationDistillationPipeline:
    def __init__(
        self,
        teacher_model: transformers.PreTrainedModel,
        student_ckpt: str,
        dataset: datasets.DatasetDict,
    ):
        if not all([split in dataset for split in ['train', 'validation', 'test']]):
            raise ValueError('Dataset must have `train`, `validation`, and `test` splits.')

        self.teacher_model = teacher_model
        self.student_ckpt = student_ckpt
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(student_ckpt)
        self.dataset = dataset
        self.num_labels = self.dataset['test'].features['labels'].num_classes
        self.id2label = self.teacher_model.config.id2label
        self.label2id = self.teacher_model.config.label2id

        tokenization_fn = self._tokenization_function(tokenizer=self.tokenizer)
        self.dataset = self.dataset.map(tokenization_fn, batched=True, remove_columns=['text'])

    def run(
        self,
        save_model_to: str,
        eval_metric: str = 'accuracy',
        optimize_hparams: bool = False,
        n_trials: int = None,
        **training_kwargs
    ):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device: {device}')
        training_args = self.default_training_args
        training_args.output_dir = save_model_to
        for k, v in training_kwargs.items():
            setattr(training_args, k, v)

        student_model = transformers.AutoModelForSequenceClassification.from_pretrained(
            self.student_ckpt
        )

        student_config = student_model.config
        student_config.update({
            'num_labels': self.num_labels,
            'id2label': self.id2label,
            'label2id': self.label2id,
        })

        trainer = KnowledgeDistillationTrainer(
            model_init=self._student_init(ckpt=self.student_ckpt, config=student_config, device=device),
            teacher=self.teacher_model,
            args=training_args,
            train_dataset=self.dataset['train'],
            eval_dataset=self.dataset['validation'],
            compute_metrics=self._compute_metrics_fn(metric=eval_metric),
            tokenizer=self.tokenizer,
        )

        if optimize_hparams:
            logging.warning(
                'Hyperparameter optimization requires large amounts of memory. You might get OOM errors.'
            )
            if n_trials is None:
                raise ValueError('Must specify `n_trials` when `optimize_hparams` is True.')

            optim_hparams = self._get_optimal_hparams(trainer, n_trials=n_trials)
            for k, v in optim_hparams.items():
                setattr(training_args, k, v)

            trainer = KnowledgeDistillationTrainer(
                model_init=self._student_init(ckpt=self.student_ckpt, config=student_config, device=device),
                teacher=self.teacher_model,
                args=training_args,
                train_dataset=self.dataset['train'],
                eval_dataset=self.dataset['validation'],
                compute_metrics=self._compute_metrics_fn(metric=eval_metric),
                tokenizer=self.tokenizer,
            )

        trainer.train()
        trainer.save_model(save_model_to)
        logging.info(f'Model saved to {save_model_to}')

    @staticmethod
    def _tokenization_function(tokenizer):
        def tokenize(batch):
            return tokenizer(batch['text'], truncation=True)
        return tokenize

    @property
    def default_training_args(self):
        return KnowledgeDistillationTrainingArguments(
            evaluation_strategy='epoch',
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            learning_rate=2e-5,
            weight_decay=0.01,
            num_train_epochs=10,
            alpha=0.125,
            temperature=7,
        )

    @staticmethod
    def _student_init(ckpt: str, config: transformers.PretrainedConfig, device: torch.device):
        def model_init():
            return transformers.AutoModelForSequenceClassification.from_pretrained(
                ckpt,
                config=config
            ).to(device)
        return model_init

    @staticmethod
    def _compute_metrics_fn(metric: str = 'accuracy'):
        metric = datasets.load_metric(metric)

        def _compute_metrics(pred):
            logits, labels = pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)
        return _compute_metrics

    @staticmethod
    def _get_optimal_hparams(trainer: transformers.Trainer, n_trials: int = 10):
        def hp_space(trial):
            return {
                'num_train_epochs': trial.suggest_int('num_train_epochs', 5, 10),
                'alpha': trial.suggest_float('alpha', 0, 1),
                'temperature': trial.suggest_int('temperature', 2, 20),
            }
        logging.info('Running hyperparameter optimization in the `maximize` direction.')
        best_run = trainer.hyperparameter_search(
            n_trials=n_trials,
            direction='maximize',
            hp_space=hp_space,
        )
        return best_run.hyperparameters
