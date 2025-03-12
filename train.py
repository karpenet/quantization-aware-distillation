import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from SegFormer import SegformerForSemanticSegmentation
from qat import QATModel
from transformers import SegformerConfig, SegformerImageProcessor
from dataclasses import dataclass
from dataset import SemanticSegmentationDataset

import evaluate
import wandb
from tqdm import tqdm


@dataclass
class TrainConfig:
    model: str = 'nvidia/segformer-b0-finetuned-ade-512-512'
    dataset: str = '../data/ADEChallengeData2016'
    num_epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 0.001
    weights_file: str = 'b0_train.pth'
    log_wandb: bool = True


class Trainer:
    def __init__(self, config):
        self.config = config

        # Define config and preprocessor
        student_config = SegformerConfig().from_pretrained(self.config.model)
        self.processor = SegformerImageProcessor(reduce_labels=True)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Create student and trainer models
        self.model = SegformerForSemanticSegmentation(config=student_config)
        self.model.to(self.device)

        # Initialize dataset and dataloader
        train_dataset = SemanticSegmentationDataset(root_dir=self.config.dataset, image_processor=self.processor)
        val_dataset = SemanticSegmentationDataset(root_dir=self.config.dataset, image_processor=self.processor, train=False)

        self.train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)

        # Define optimizer and Loss Fn
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        self.criterion = CrossEntropyLoss(ignore_index=255)

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
            self.model.train()
            for batch_idx, batch in enumerate(tqdm(self.train_loader, desc='Batch', position=1, leave=False, unit='batch')):
                images = batch['pixel_values'].to(self.device)
                labels = batch['labels'].to(self.device)

                output = self.model(images)
                upsampled_logits = nn.functional.interpolate(
                    output.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
                )
                
                loss = self.criterion(upsampled_logits, labels)

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

        torch.save(self.model.state_dict(), "saved_weights/" + self.config.weights_file)

    def test(self):
        # Define metric
        metric = evaluate.load('mean_iou')

        # Validation Loop
        self.model.eval()
        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc='Validation', position=1, leave=False, unit='batch')):
            images = batch['pixel_values'].to(self.device)
            labels = batch['labels'].to(self.device)

            with torch.no_grad():
                output = self.model(images)
                upsampled_logits = nn.functional.interpolate(output.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
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
    config = TrainConfig(batch_size=4, num_epochs=50, log_wandb=True)
    trainer = Trainer(config)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        torch.save(trainer.model.state_dict(), "saved_weights/" + config.weights_file)