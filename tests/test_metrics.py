import torch
from banana_creda.utils.metrics import MetricTracker

def test_compute_accuracy():
    preds = torch.tensor([0, 1, 2, 0, 1])
    labels = torch.tensor([0, 1, 0, 0, 2])
    # Matches: (0=0), (1=1), (2=0)X, (0=0), (1=2)X
    # Matches are 3 out of 5: 0.6
    
    num_classes = 3
    overall_acc, per_class_acc = MetricTracker.compute_accuracy(preds, labels, num_classes)
    
    assert overall_acc == 0.6
    # Class 0: 2 correct out of 3. (pred[0,2,3], label[0,2,3]) -> label[0]==0 (match), label[2]==0 (pred is 2, no match), label[3]==0 (match)
    # Correct = 2. Total = 3. Acc = 2/3 = 0.666...
    assert pytest.approx(per_class_acc[0], 0.001) == 2/3
    # Class 1: 1 correct out of 0? No, wait. 
    # Label 1 exists at idx 1. Pred matches at idx 1.
    # Correct = 1. Total = 1. Acc = 1.0
    assert per_class_acc[1] == 1.0
    # Class 2: Correct = 0. Total = 1. Acc = 0.0
    assert per_class_acc[2] == 0.0

def test_compute_full_metrics():
    preds = torch.tensor([0, 1, 2, 0, 1])
    labels = torch.tensor([0, 1, 0, 0, 2])
    num_classes = 3
    device = torch.device("cpu")
    
    results = MetricTracker.compute_full_metrics(preds, labels, num_classes, device)
    
    assert results["overall_acc"] == 0.6
    assert len(results["per_class"]) == 3
    # Check Class 0 Recall (Sensitivity)
    # TP=2, FN=1 (at idx 2 label is 0 pred is 2)
    # Recall = TP/(TP+FN) = 2/3
    assert pytest.approx(results["per_class"][0]["recall"], 0.001) == 2/3
    # Check Class 0 Precision
    # TP=2, FP=0 (preds are at idx 0 and 3, both labels are 0)
    # Precision = TP/(TP+FP) = 2/2 = 1.0
    assert results["per_class"][0]["precision"] == 1.0

import pytest
