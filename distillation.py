import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
import torch.nn.functional as F

from SegFormer import SegformerForSemanticSegmentation
from qat import QATModel
from transformers import SegformerConfig, SegformerImageProcessor
from dataclasses import dataclass
from dataset import SemanticSegmentationDataset

import evaluate
import wandb
from tqdm import tqdm


@dataclass
class DistillationConfig:
    trainer_model: str = 'nvidia/segformer-b4-finetuned-ade-512-512'
    student_model: str = 'nvidia/segformer-b0-finetuned-ade-512-512'
    dataset: str = '../data/ADEChallengeData2016'
    num_epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 0.001
    temperature: float = 3.0
    alpha: float = 0.5
    weights_file: str = 'b4_to_b0_distill.pth'
    log_wandb: bool = True
    qat: bool = False


class DistillationLoss(nn.Module):
    def __init__(self, temperature=3.0, alpha=0.5):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha

        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255)
    
    def forward(self, student_logits, teacher_logits, labels):
        soft_targets = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)

        upsampled_logits = nn.functional.interpolate(soft_targets, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        upsampled_logits = upsampled_logits.argmax(dim=1)

        kl_loss = self.kl_loss(soft_targets, soft_teacher)
        ce_loss = self.ce_loss(upsampled_logits.float(), labels.long())

        return self.alpha * ce_loss + (1 - self.alpha) * kl_loss


class DistilModel:
    def __init__(self, config):
        self.config = config

        # Define config and preprocessor
        student_config = SegformerConfig().from_pretrained(self.config.student_model)
        self.processor = SegformerImageProcessor(reduce_labels=True)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Create student and trainer models
        self.trainer_model = SegformerForSemanticSegmentation.from_pretrained(self.config.trainer_model)
        self.student_model = SegformerForSemanticSegmentation(config=student_config)

        if self.config.qat:
            self.student_model = QATModel(self.student_model)

        self.trainer_model.to(self.device)
        self.student_model.to(self.device)
        self.trainer_model.eval()

        # Initialize dataset and dataloader
        train_dataset = SemanticSegmentationDataset(root_dir=self.config.dataset, image_processor=self.processor)
        val_dataset = SemanticSegmentationDataset(root_dir=self.config.dataset, image_processor=self.processor, train=False)

        self.train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)

        # Define optimizer
        self.optimizer = optim.AdamW(self.student_model.parameters(), lr=self.config.learning_rate)

        # Define loss
        self.criterion = DistillationLoss(temperature=self.config.temperature, alpha=self.config.alpha)

        if self.config.log_wandb:
            wandb.init(
                project="Segformer-Distillation",
                config=self.config
            )

    def __del__(self):
        if self.config.log_wandb:
            wandb.finish()        

    def train(self):
        for epoch in tqdm(range(self.config.num_epochs), desc='Epoch', position=0, unit='epoch'):
            total_loss = 0.0
            self.student_model.train()
            for batch_idx, batch in enumerate(tqdm(self.train_loader, desc='Batch', position=1, leave=False, unit='batch')):
                images = batch['pixel_values'].to(self.device)
                labels = batch['labels'].to(self.device)

                with torch.no_grad():
                    trainer_out = self.trainer_model(images)
                student_out = self.student_model(images)

                # Compute distillation loss
                loss = self.criterion(student_out.logits, trainer_out.logits, labels)

                # Step training
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                if batch_idx % 10 == 0:
                    if self.config.log_wandb:
                        wandb.log({"Distillation Loss": loss.item()})

            torch.cuda.empty_cache()

            avg_loss = total_loss / len(self.train_loader)

            validation_metrics = self.test()
            training_metrics = {"Epoch": epoch, " Train Loss": avg_loss}
        
            if self.config.log_wandb:    
                wandb.log({**training_metrics, **validation_metrics})

        if self.config.qat:
            QATModel.convert_fully_quantized(self.student_model)

        torch.save(self.student_model.state_dict(), "saved_weights/" + self.config.weights_file)

    def test(self):
        # Define metric
        metric = evaluate.load('mean_iou')

        # Validation Loop
        self.student_model.eval()
        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc='Validation', position=1, leave=False, unit='batch')):
            images = batch['pixel_values'].to(self.device)
            labels = batch['labels'].to(self.device)

            with torch.no_grad():
                output = self.student_model(images)
                logits = output.logits
                upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                predicted = upsampled_logits.argmax(dim=1)

            metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())

        # Compute metrics
        result = metric.compute(
            predictions=predicted.cpu(),
            references=labels.cpu(),
            num_labels=150,
            ignore_index=255,
            nan_to_num=0,
            reduce_labels=False,
        )

        torch.cuda.empty_cache()

        return result


if __name__ == "__main__":
    config = DistillationConfig(batch_size=8, num_epochs=50, log_wandb=True)
    distillation = DistilModel(config)
    
    try:
        distillation.train()
    except KeyboardInterrupt:
        torch.save(distillation.student_model.state_dict(), "saved_weights/" + config.weights_file)