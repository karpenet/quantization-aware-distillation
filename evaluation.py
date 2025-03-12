import evaluate
import torch
from dataset import SemanticSegmentationDataset
from transformers import SegformerImageProcessor
from SegFormer import SegformerForSemanticSegmentation
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import json
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict

import numpy as np
from ade20k_utils import ade_palette


class Metrics:
    @staticmethod
    def get_dataloader(root_dir, batch_size=8):
        processor = SegformerImageProcessor(reduce_labels=True)
        dataset = SemanticSegmentationDataset(root_dir=root_dir, image_processor=processor, train=False)
        
        dataloader = DataLoader(dataset, batch_size=batch_size)

        return dataloader

    @staticmethod
    def save_class_frequency(class_freqs):
        plt.bar(range(len(class_freqs)), class_freqs, width=0.6)
        plt.title("Class Frequency")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.savefig('class_freq.jpg')
        plt.clf()

    @staticmethod
    def save_figures(results):
        bins = results['per_category_iou']
        plt.bar(range(len(bins)), bins, width=0.6)
        plt.title("Per Category Iou")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.savefig('per_category_iou_pre.jpg')
        plt.clf()

        bins = results['per_category_accuracy']
        plt.bar(range(len(bins)), bins, width=0.6)
        plt.title("Per Category Accuracy")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.savefig('per_category_accuracy_pre.jpg')
        plt.clf()

    @staticmethod
    def save_outputs(images, labels, output, processor, batch_idx=0, batch_size=0):
        for idx in range(len(images)):
            predicted_segmentation_map = output[idx].cpu().numpy()
            color_seg = np.zeros((predicted_segmentation_map.shape[0],
                predicted_segmentation_map.shape[1], 3), 
                dtype=np.uint8
            ) # height, width, 3

            label_seg  = np.zeros((predicted_segmentation_map.shape[0],
                predicted_segmentation_map.shape[1], 3), 
                dtype=np.uint8
            ) # height, width, 3

            palette = np.array(ade_palette())
            for label, color in enumerate(palette):
                color_seg[predicted_segmentation_map == label, :] = color
                label_seg[labels[idx] == label, :] = color

            # Convert to BGR
            color_seg = color_seg[..., ::-1]
            label_seg = label_seg[..., ::-1]

            seg = np.concatenate((label_seg, color_seg), axis=1)
            img = np.concatenate((np.moveaxis(images[idx], 0, -1), np.moveaxis(images[idx], 0, -1)), axis=1)

            # Show image + mask
            img = img + seg * 0.5
            img = img.astype(np.uint8)

            img = Image.fromarray(img)
            img.save(f"output/{batch_idx * batch_size + idx}.jpeg")

    @staticmethod
    @torch.no_grad()
    def evaluate_model(
        model: nn.Module,
        dataset: str,
        save_output: bool = False,
        save_fig: bool = False,
        save_class_frequency: bool = False,
    ):
        counter = defaultdict(int)
        if save_output:
            processor = SegformerImageProcessor(do_resize=False)

        id2label = json.load(open('id2label.json', "r"))
        id2label = {int(k): v for k, v in id2label.items()}

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        batch_size = 8
        dataloader = Metrics.get_dataloader(dataset, batch_size=batch_size)
        metric = evaluate.load('mean_iou')
        model.to(device)

        model.eval()
        for idx, batch in enumerate(tqdm(dataloader, desc="Evaluating", unit="batch")):
            images = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            if save_class_frequency:
                unique, counts = np.unique(labels.detach().cpu().numpy(), return_counts=True)
                for u, c in zip(unique, counts):
                    counter[u] += c

                class_counts = np.array(counts)
                class_freqs = class_counts / class_counts.sum()

            with torch.no_grad():
                output = model(images)
                logits = output.logits
                upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                predicted = upsampled_logits.argmax(dim=1)

            if save_output:
                Metrics.save_outputs(
                    images=images.detach().cpu().numpy(), 
                    labels=labels.detach().cpu().numpy(), 
                    output=predicted, 
                    processor=processor, 
                    batch_idx=idx, 
                    batch_size=batch_size,
                )

            metric.add_batch(
                predictions=predicted.detach().cpu().numpy(), 
                references=labels.detach().cpu().numpy(),
            )

        result = metric.compute(
            predictions=predicted.cpu(),
            references=labels.cpu(),
            num_labels=len(id2label),
            ignore_index=255,
            nan_to_num=0.,
            reduce_labels=False,
        )

        if save_class_frequency:
            Metrics.save_class_frequency(class_freqs)

        if save_fig:
            Metrics.save_figures(results)

        return result


if __name__ == "__main__":
    model = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512')
    model.load_state_dict(torch.load('saved_weights/b4_to_b0_distill.pth', weights_only=True))
    results = Metrics.evaluate_model(model, dataset='../data/ADEChallengeData2016', save_fig=False, save_output=True)
    print(results)

