## author: xin luo
## create: 2021.10.27
## des: batch-based accuracy evaluation.


import torch

## -- batch-based; torch-based -- ##
## ------------------------------ ##
def oa_binary(pred, truth):
    ''' des: calculate overall accuracy (2-class classification) for each batch
        input: 
            pred(4D tensor), and truth(4D tensor)
    '''
    pred_bi = torch.where(pred>0.5, 1., 0.)   # [N,C,H,W]
    inter = pred_bi+truth
    area_inter = torch.histc(inter.float(), bins=3, min=0, max=2)
    area_inter = area_inter[0:3:2]
    area_pred = torch.histc(pred, bins=2, min=0, max=1)
    oa = area_inter/(area_pred+0.0000001)
    oa = oa.mean()
    return oa

def miou_binary(pred, truth):
    ''' des: calculate miou (2-class classification) for each batch
        input: 
            pred(4D tensor), and truth(4D tensor)
    '''
    pred_bi = torch.where(pred>0.5, 1., 0.)   # [N,C,H,W]
    inter = pred_bi+truth
    area_inter = torch.histc(inter.float(), bins=3, min=0, max=2)
    area_inter = area_inter[0:3:2]
    area_pred = torch.histc(pred, bins=2, min=0, max=1)
    area_truth = torch.histc(truth.float(), bins=2, min=0, max=1)
    area_union = area_pred + area_truth - area_inter
    iou = area_inter/(area_union+0.0000001)
    miou = iou.mean()
    return miou

def oa_multi(pred, truth, num_classes=3):
    """
    Calculate Overall Accuracy (OA) for multi-class classification.
    Args:
        pred (4D tensor): Predicted probabilities [N, C, H, W].
        truth (3D tensor): Ground truth labels [N, H, W].
        num_classes (int): Total number of classes (default is 3).
    Returns:
        oa (float): Overall accuracy for the batch.
    """
    # Convert predicted probabilities to class predictions
    pred_labels = torch.argmax(pred, dim=1)  # [N, H, W]

    # Compare predictions with ground truth
    correct = (pred_labels == truth).float()  # Binary mask for correct predictions
    total_correct = torch.sum(correct)  # Total number of correct pixels
    total_pixels = truth.numel()  # Total number of pixels

    # Compute overall accuracy
    oa = total_correct / (total_pixels + 1e-7)  # Avoid division by zero
    return oa


def miou_multi(pred, truth, num_classes=3):
    """
    Calculate Mean Intersection over Union (mIoU) for multi-class classification.
    Args:
        pred (4D tensor): Predicted probabilities [N, C, H, W].
        truth (3D tensor): Ground truth labels [N, H, W].
        num_classes (int): Total number of classes (default is 3).
    Returns:
        miou (float): Mean IoU for the batch.
    """
    # Convert predicted probabilities to class predictions
    pred_labels = torch.argmax(pred, dim=1)  # [N, H, W]

    iou_per_class = []
    for class_idx in range(num_classes):
        # Create binary masks for the current class
        pred_class = (pred_labels == class_idx).float()  # Predicted pixels for this class
        truth_class = (truth == class_idx).float()  # Ground truth pixels for this class

        # Compute intersection and union
        intersection = torch.sum(pred_class * truth_class)
        union = torch.sum(pred_class) + torch.sum(truth_class) - intersection

        # Handle case where there are no pixels for this class in truth or predictions
        if union == 0:
            iou = torch.tensor(1.0, device=pred.device)  # Perfect IoU if there are no pixels
        else:
            iou = intersection / (union + 1e-7)  # Avoid division by zero

        iou_per_class.append(iou)

    # Compute mean IoU across all classes
    miou = torch.mean(torch.stack(iou_per_class))
    return miou