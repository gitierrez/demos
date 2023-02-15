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
    # rename columns
    if text_column != 'text':
        dataset = dataset.rename_column(text_column, 'text')
    if label_column != 'labels':
        dataset = dataset.rename_column(label_column, 'labels')

    # check dataset only has 'train', 'validation', and 'test' splits
    if set(dataset.column_names) != {'train', 'validation', 'test'}:
        raise ValueError(
            'Dataset must only have `train`, `validation` and `test` columns. '
            f'Found {dataset.column_names} instead.'
        )

    # check dataset has 'text' and 'labels' columns
    for split in dataset:
        if 'text' not in dataset[split].column_names:
            raise ValueError(
                f'`dataset[{split}]` must have a `text` column. If it has a different name '
                f'in your dataset please specify it using the `text_column` param. '
                f'For example: distill_model(..., text_column="sentences")'
            )
        if 'labels' not in dataset[split].column_names:
            raise ValueError(
                f'`dataset[{split}]` must have a `labels` column. If it has a different name '
                f'in your dataset please specify it using the `label_column` param. '
                f'For example: distill_model(..., label_column="intents")'
            )

    # distill the model
    TextClassificationDistillationPipeline.run(
        teacher_model=teacher_model,
        student_ckpt=student_ckpt,
        dataset=dataset,
        save_model_to=save_model_to,
        eval_metric=evaluation_metric,
        optimize_hparams=optimize_hparams,
        n_trials=n_trials,
        **training_kwargs
    )
    return transformers.AutoModelForSequenceClassification.from_pretrained(save_model_to)
