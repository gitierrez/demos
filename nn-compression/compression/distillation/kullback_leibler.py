import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers


class KnowledgeDistillationTrainingArguments(transformers.TrainingArguments):
    """
    Training arguments for knowledge distillation.

    Args:
        alpha: The importance given to the cross-entropy loss.
        temperature: The temperature used in the softmax function.
    """
    def __init__(self, *args, alpha: float = 0.5, temperature: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        if alpha < 0 or alpha > 1:
            raise ValueError('alpha must be in the range [0, 1]')
        self.alpha = alpha
        self.temperature = temperature


class KnowledgeDistillationTrainer(transformers.Trainer):
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
        loss_kld = nn.functional.kl_div(
            input=F.log_softmax(student_logits / self.args.temperature, dim=-1),
            target=F.softmax(teacher_logits / self.args.temperature, dim=-1),
            reduction='batchmean'
        )
        # re-scale the loss to match the scale of the cross-entropy loss
        loss_kld = (self.args.temperature ** 2) * loss_kld
        student_loss = self.args.alpha * loss_ce + (1.0 - self.args.alpha) * loss_kld
        return (student_loss, student_outputs) if return_outputs else student_loss
