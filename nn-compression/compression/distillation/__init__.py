import transformers
import datasets

from .distillation import TextClassificationDistillationPipeline


def distill_model(
    teacher_model: transformers.PreTrainedModel,
    student_ckpt: str,
    dataset: datasets.DatasetDict,
    save_model_to: str,
    text_column: str = 'text',
    label_column: str = 'labels',
    evaluation_metric: str = 'accuracy',
    optimize_hparams: bool = False,
    n_trials: int = None,
    **training_kwargs
):
    if text_column != 'text':
        dataset = dataset.rename_column(text_column, 'text')
    if label_column != 'labels':
        dataset = dataset.rename_column(label_column, 'labels')
    # distill the model
    distillation_pipeline = TextClassificationDistillationPipeline(
        teacher_model=teacher_model,
        student_ckpt=student_ckpt,
        dataset=dataset,
    )
    distillation_pipeline.run(
        save_model_to=save_model_to,
        eval_metric=evaluation_metric,
        optimize_hparams=optimize_hparams,
        n_trials=n_trials,
        **training_kwargs
    )
    return transformers.AutoModelForSequenceClassification.from_pretrained(save_model_to)
