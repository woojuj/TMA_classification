**train_val_loss.svg**

This plot shows the change in loss for the training and validation datasets over epochs. Training loss starts high and gradually decreases, indicating that the model is learning from the data. In contrast, validation loss stabilizes or slightly increases as training progresses. For example, training loss reduced to a minimum of 0.12, while validation loss stabilized around 0.45.

**metrics_over_epochs.svg**

This plot visualizes how many metrics, such as Accuracy, F1 Score, Precision, and Recall, change across epochs during training. Accuracy on validation set started at around 60%, gradually improved, and reached 85% by the end of training. Similarly, F1 Score, Precision, and Recall followed a rising trend, indicating consistent improvement in performance across both training and validation datasets.

**roc_curve_test.svg**

This graph illustrates the ROC curve for the test dataset. It visualizes the relationship between True Positive Rate (TPR) and False Positive Rate (FPR) for each class. The Area Under the Curve (AUC) quantifies the model's classification performance. For instance, the AUC for class benign and for class tumor are 0.98, both indicating high classification performance. A curve farther from the diagonal indicates better performance.

**confusion_roc.svg**

This visualization combines the Confusion Matrix and ROC curves for training and validation datasets. For the training dataset, there were 3307 True Positives (TP) and 1110 True Negatives (TN), reflecting high accuracy. The validation dataset showed slightly lower TP values. Additionally, the ROC curves for both datasets achieved AUC scores above 0.95 for each class, demonstrating robust classification performance.

**confusion_matrix.svg**

This graph represents the Confusion Matrix for the test dataset. True Positives (TP) were 804, True Negatives (TN) were 261, False Positives (FP) were 34, and False Negatives (FN) were 42. These results suggest that the model maintained stable performance on the test set, with an overall accuracy of approximately 93%.

**Output_tma.out**

This output file summarizes the training and evaluation process of an image classification model on the TMA dataset. It includes dataset distribution, training, validation and test process and performance. While the training was initially set for 30 epochs, Early Stopping was triggered at epoch 18 due to no improvement in validation loss for 8 epochs. The initial learning rate was 0.001 and progressively decreased, reaching as low as 1.0000000000000004e-08 by the end.
