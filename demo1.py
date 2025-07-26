import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from d2l import torch as d2l
from MATFLDA import *
from data_train_test_1 import *

def save_roc_pr_data(fpr, tpr, precision, recall, dataset_name, fold):
    # DataFrame
    roc_df = pd.DataFrame({
        'FPR': fpr,
        'TPR': tpr
    })
    pr_df = pd.DataFrame({
        'Precision': precision,
        'Recall': recall
    })

    # CSV
    roc_df.to_csv(f'{dataset_name}_fold{fold}_roc_data.csv', index=False)
    pr_df.to_csv(f'{dataset_name}_fold{fold}_pr_data.csv', index=False)

def demo(dataset, fold, device):
    tt_set = prep_dataloader(dataset=dataset, fold=fold, mode='test', dimension=3, batch_size=32)

    model = torch.load('model/demo_' + dataset + '_fold' + str(fold) + '.pth', map_location=device)
    labels, preds = test(tt_set, model, device)

    # ROC Curve
    plt.figure()
    fpr, tpr, _ = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot(fpr, tpr, label=f'LDAformer (AUC={roc_auc:.3f})')
    plt.legend()

    # PR Curve
    plt.figure()
    precision, recall, _ = precision_recall_curve(labels, preds)
    pr_auc = auc(recall, precision)
    plt.title('PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.plot(recall, precision, label=f'LDAformer (AUPR={pr_auc:.3f})')
    plt.legend()


    save_roc_pr_data(fpr, tpr, precision, recall, dataset, fold)


#    plt.savefig(f"{dataset}_fold{fold}_ROC.png")
#    plt.savefig(f"{dataset}_fold{fold}_PR.png")

    plt.show()

if __name__ == "__main__":
    dataset = 'dataset2'
    fold = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    demo(dataset, fold, device)
