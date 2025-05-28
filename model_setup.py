import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B7_Weights,
    ConvNeXt_Tiny_Weights,
    ConvNeXt_Base_Weights
)


from torchmetrics import Accuracy, Precision, Recall, AUROC, F1Score
from tqdm.auto import tqdm
from typing import Dict, List, Tuple



def call_model(model_name='convnext_tiny', device=None, fine_tune=None):
    """
    Initialize ConvNeXt or EfficientNet using torchvision models.

    Args:
        model_name: One of ['convnext_tiny', 'convnext_base', 'efficientnet_b0', 'efficientnet_b7']
        device: torch.device
        fine_tune: None (frozen), 'last_two' (last two blocks), or 'all' (entire model)
    """
    # Device setup
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model initialization with pretrained weights
    if model_name.startswith('convnext'):
        if model_name == 'convnext_tiny':
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            model = models.convnext_tiny(weights=weights)
        elif model_name == 'convnext_base':
            weights = ConvNeXt_Base_Weights.IMAGENET1K_V1
            model = models.convnext_base(weights=weights)
        else:
            raise ValueError(f"Unsupported ConvNeXt variant: {model_name}")

    elif model_name.startswith('efficientnet'):
        if model_name == 'efficientnet_b0':
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            model = models.efficientnet_b0(weights=weights)
        elif model_name == 'efficientnet_b7':
            weights = EfficientNet_B7_Weights.IMAGENET1K_V1
            model = models.efficientnet_b7(weights=weights)
        else:
            raise ValueError(f"Unsupported EfficientNet variant: {model_name}")
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Freezing parameters
    if fine_tune is None:
        for param in model.parameters():
            param.requires_grad = False
    elif fine_tune == 'last_two':
        # Unfreeze last two blocks
        if model_name.startswith('convnext'):
            for block in model.features[-2:]:
                for param in block.parameters():
                    param.requires_grad = True
        else:  # EfficientNet
            for layer in model.features[-2:]:
                for param in layer.parameters():
                    param.requires_grad = True
    elif fine_tune == 'all':
        for param in model.parameters():
            param.requires_grad = True
    else:
        raise ValueError("fine_tune must be None, 'last_two', or 'all'")

    # Replace classifier head for binary classification
    if model_name.startswith('convnext'):
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            model.classifier[0],  # Keep LayerNorm2d
            model.classifier[1],  # Keep AdaptiveAvgPool2d
            nn.Flatten(),
            nn.Linear(in_features, 1),

        )
    else:  # EfficientNet
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, 1)

        )



    return model


def train_step(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
) -> Tuple[float, float]:
    """Train step with TorchMetrics for accuracy calculation."""
    model.train()
    train_loss = 0.0

    # Initialize TorchMetrics Accuracy (binary classification)
    metric = Accuracy(task="binary").to(device)

    for images, labels, _ in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        # Handle multi-GPU/DP outputs
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        # Loss computation
        loss = loss_fn(outputs, labels.unsqueeze(1).float())
        loss.backward()
        optimizer.step()

        # Update metrics (FIXED: squeeze predictions to match labels)
        probs = torch.sigmoid(outputs).squeeze(1)  # Shape: [batch_size]
        metric.update(probs, labels)

        # Update loss
        train_loss += loss.item() * images.size(0)

    # Compute final metrics
    avg_loss = train_loss / len(dataloader.dataset)
    avg_acc = metric.compute().item()
    metric.reset()

    scheduler.step()
    return avg_loss, avg_acc



def test_step(model: nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: nn.Module,
              device: torch.device) -> Tuple[float, float, float, float, float, float]:
    """Test step with proper type handling and metric reset"""
    model.eval()

    # Initialize metrics
    metrics = {
        'acc': Accuracy(task='binary').to(device),
        'precision': Precision(task='binary').to(device),
        'recall': Recall(task='binary').to(device),
        'auc': AUROC(task='binary').to(device),
        'f1': F1Score(task='binary').to(device)
    }

    test_loss = 0.0

    try:
        with torch.no_grad():
            for images, labels, _ in dataloader:
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                # Calculate loss (ensure proper types)
                loss = loss_fn(outputs, labels.float().unsqueeze(1))
                test_loss += loss.item() * images.size(0)

                # Get probabilities and ensure proper shapes
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()

                # Reshape if needed
                if len(labels.shape) == 1:
                    labels = labels.unsqueeze(1)
                if len(preds.shape) == 1:
                    preds = preds.unsqueeze(1)

                # Update metrics
                for metric in metrics.values():
                    metric.update(preds, labels)

        # Compute final metrics (ensure float conversion)
        avg_loss = float(test_loss / len(dataloader.dataset))
        results = {
            'loss': avg_loss,
            'acc': float(metrics['acc'].compute().item()) ,
            'precision': float(metrics['precision'].compute().item()),
            'recall': float(metrics['recall'].compute().item()),
            'auc': float(metrics['auc'].compute().item()),
            'f1': float(metrics['f1'].compute().item())
        }

        return (
            results['loss'],
            results['acc'],
            results['precision'],
            results['recall'],
            results['auc'],
            results['f1']
        )

    finally:
        # Reset metrics
        for metric in metrics.values():
            metric.reset()


def summarize_kfold_metrics(foldperf):
    """Comprehensive metric summary"""
    metrics = {
        'acc': [], 'precision': [], 'recall': [],
        'auc': [], 'f1': []
    }

    print("\n=== Comprehensive Fold Performance ===")
    for fold in sorted(foldperf.keys()):
        fold_metrics = {
            'acc': foldperf[fold]['test_acc'][-1],
            'precision': foldperf[fold]['test_precision'][-1],
            'recall': foldperf[fold]['test_recall'][-1],
            'auc': foldperf[fold]['test_auc'][-1],
            'f1': foldperf[fold]['test_f1'][-1]
        }

        print(f"\n{fold}:")
        print(f"Accuracy: {fold_metrics['acc']*100:.2f}%")
        print(f"Precision: {fold_metrics['precision']:.4f}")
        print(f"Recall: {fold_metrics['recall']:.4f}")
        print(f"AUC: {fold_metrics['auc']:.4f}")
        print(f"F1: {fold_metrics['f1']:.4f}")

        for k in metrics.keys():
            metrics[k].append(fold_metrics[k])

    print("\n=== Aggregate Metrics ===")
    for metric, values in metrics.items():
        unit = '%' if metric == 'acc' else ''
        print(f"Mean {metric.capitalize()}: {np.mean(values):.4f}{unit} ± {np.std(values):.4f}")

    return metrics

