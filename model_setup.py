import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B7_Weights,
    ConvNeXt_Tiny_Weights,
    ConvNeXt_Base_Weights
)

from sklearn.model_selection import StratifiedKFold  
from torchmetrics import Accuracy, Precision, Recall, AUROC, F1Score
from tqdm.auto import tqdm
from typing import Dict, List, Tuple



import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B7_Weights,
    ConvNeXt_Tiny_Weights,
    ConvNeXt_Base_Weights
)

def call_model(model_name='convnext_tiny', device=None, fine_tune=None):
    """
    Initialize ConvNeXt or EfficientNet using torchvision models.

    Args:
        model_name: One of ['convnext_tiny', 'convnext_base', 'efficientnet_b0', 'efficientnet_b7']
        device: torch.device
        fine_tune: None (frozen), 'head_only' (final conv + classifier), 
                   'last_two' (last two blocks), or 'all' (entire model)
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
            
        # For ConvNeXt, identify the final convolutional layer
        final_conv = model.features[-1].block[-1]

    elif model_name.startswith('efficientnet'):
        if model_name == 'efficientnet_b0':
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            model = models.efficientnet_b0(weights=weights)
        elif model_name == 'efficientnet_b7':
            weights = EfficientNet_B7_Weights.IMAGENET1K_V1
            model = models.efficientnet_b7(weights=weights)
        else:
            raise ValueError(f"Unsupported EfficientNet variant: {model_name}")
            
        # For EfficientNet, identify the final convolutional layer
        final_conv = model.features[-1]
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Freezing parameters based on fine_tune option
    if fine_tune is None:
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
            
    elif fine_tune == 'head_only':
        # Freeze all parameters first
        for param in model.parameters():
            param.requires_grad = False
            
        # Unfreeze the final convolutional layer
        for param in final_conv.parameters():
            param.requires_grad = True
            
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
        # All parameters remain trainable
        pass
    else:
        raise ValueError("fine_tune must be None, 'head_only', 'last_two', or 'all'")

    # Replace classifier head for binary classification
    if model_name.startswith('convnext'):
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            model.classifier[0],  # Keep LayerNorm2d
            model.classifier[1],  # Keep AdaptiveAvgPool2d
            nn.Flatten(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, 1)
        )
    else:  # EfficientNet
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
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




def train_kfold_model(catheter_predictions,
                     atria_predictions,
                     labels,
                     ids=None,
                     original_images=None,
                     size=600,
                     model_name='efficientnet_b7',
                     num_epochs=60, 
                     batch_size=8,
                     k=5, 
                     fine_tune='last_two'):
    """Stratified K-fold training with tqdm progress bars"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    IMG_SIZE = size

    # Define your transforms (unchanged)
    train_transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE, interpolation=cv2.INTER_NEAREST),
        A.HorizontalFlip(p=0.3),
        A.Rotate(limit=15, p=0.2, border_mode=cv2.BORDER_CONSTANT),
        A.OneOf([
            A.ElasticTransform(alpha=30, sigma=5, p=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.15, 
                           interpolation=cv2.INTER_NEAREST,
                           border_mode=cv2.BORDER_CONSTANT, p=0.3)
        ], p=0.3),
        A.ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE, interpolation=cv2.INTER_NEAREST),
        A.ToTensorV2()
    ])
  
    # Use StratifiedKFold instead of KFold
    splits = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    foldperf = {}

    # Convert labels to numpy array for stratified splitting
    labels_np = np.array(labels)
    
    # StratifiedKFold.split needs both features (X) and labels (y)
    # We'll use indices as dummy features since we have separate components
    dummy_X = np.arange(len(labels_np))
    
    for fold, (train_idx, val_idx) in enumerate(splits.split(dummy_X, labels_np)):
        print(f'\n=== Fold {fold + 1}/{k} ===')
        print(f"Train class distribution: {np.bincount(labels_np[train_idx])}")
        print(f"Val class distribution: {np.bincount(labels_np[val_idx])}")

        # Split raw data components using stratified fold indices
        train_catheter = catheter_predictions[train_idx]
        train_atria = atria_predictions[train_idx]
        train_labels = [labels[i] for i in train_idx]
        train_ids = [ids[i] for i in train_idx] if ids else None
        train_originals = original_images[train_idx] if original_images else None

        val_catheter = catheter_predictions[val_idx]
        val_atria = atria_predictions[val_idx]
        val_labels = [labels[i] for i in val_idx]
        val_ids = [ids[i] for i in val_idx] if ids else None
        val_originals = original_images[val_idx] if original_images else None

        # Create datasets with appropriate transforms
        train_dataset = ClassificationDataset(
            catheter_predictions=train_catheter,
            atria_predictions=train_atria,
            labels=train_labels,
            ids=train_ids,
            original_images=train_originals,
            transform=train_transform
        )
        
        val_dataset = ClassificationDataset(
            catheter_predictions=val_catheter,
            atria_predictions=val_atria,
            labels=val_labels,
            ids=val_ids,
            original_images=val_originals,
            transform=val_transform
        )

        # Create dataloaders with stratified batches
        effective_batch = batch_size * max(1, num_gpus)
        
        # For training, we can use StratifiedSampler if needed (optional)
        train_loader = DataLoader(train_dataset, batch_size=effective_batch, 
                                shuffle=True, pin_memory=True, num_workers=2)
        test_loader = DataLoader(val_dataset, batch_size=effective_batch,
                               shuffle=False, pin_memory=True, num_workers=2)

        # Rest of your training code remains the same...
        model = call_model(model_name=model_name, device='cpu', fine_tune=fine_tune)
        if num_gpus > 1:
            model = nn.DataParallel(model)
        model = model.to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                        lr=0.0001, weight_decay=0.001)

        # LR scheduling
        warmup_scheduler = LinearLR(optimizer, start_factor=0.01, 
                                  end_factor=1.0, total_iters=5)
        cosine_scheduler = CosineAnnealingLR(optimizer, 
                                           T_max=num_epochs-5, eta_min=1e-6)
        scheduler = SequentialLR(optimizer, 
                               schedulers=[warmup_scheduler, cosine_scheduler],
                               milestones=[5])

        history = {
            'train_loss': [], 'test_loss': [],
            'train_acc': [], 'test_acc': [],
            'test_precision': [], 'test_recall': [],
            'test_auc': [], 'test_f1': []
        }

        for epoch in tqdm(range(num_epochs), desc=f'Epochs'):
            free_gpu_memory()

            # Training
            train_loss, train_acc = train_step(
                model, train_loader, criterion, optimizer, scheduler, device)

            # Validation with all metrics
            (test_loss, test_acc, test_precision,
             test_recall, test_auc, test_f1) = test_step(
                model, test_loader, criterion, device)

            # Store results
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            history['test_precision'].append(test_precision)
            history['test_recall'].append(test_recall)
            history['test_auc'].append(test_auc)
            history['test_f1'].append(test_f1)

            # Enhanced progress reporting
            if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == num_epochs:
                print(f"Epoch {epoch+1:03d} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Train Acc: {train_acc:.2f}% | "
                      f"Test Loss: {test_loss:.4f}\n"
                      f"Test Metrics: "
                      f"Acc: {test_acc*100:.2f}% | "
                      f"Precision: {test_precision:.4f} | "
                      f"Recall: {test_recall:.4f} | "
                      f"AUC: {test_auc:.4f} | "
                      f"F1: {test_f1:.4f}")

        foldperf[f'fold{fold+1}'] = history

    return foldperf

