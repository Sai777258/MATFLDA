import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve, average_precision_score, accuracy_score, precision_score, recall_score, f1_score, auc
from MATFLDA import LDAformer
from data_train_test_1 import prep_dataloader, train, test

# Config

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config = {
    'dataset': 'dataset2',
    'fold': 0,
    'n_epochs':10,
    'batch_size': 32,
    'dimension': 3,
    'd_model': 24,
    'agent_n_heads': 8,
    'local_n_heads': 4,
    'e_layers': 5,
    'd_ff':2,
    'agent_num':1
}

tr_set = prep_dataloader(dataset=config['dataset'], fold=config['fold'], mode='train', dimension=config['dimension'], batch_size=config['batch_size'])
tt_set = prep_dataloader(dataset=config['dataset'], fold=config['fold'], mode='test', dimension=config['dimension'], batch_size=config['batch_size'])

seq_len = tr_set.dataset[0][0].shape[0]
d_input = tr_set.dataset[0][0].shape[1]

while True:
    model = LDAformer(
        seq_len=seq_len,
        d_input=d_input,
        d_model=config['d_model'],
        agent_n_heads=config['agent_n_heads'],
        local_n_heads=config['local_n_heads'],
        e_layers=config['e_layers'],
        d_ff=config['d_ff'],  # mlp_ratio
        agent_num=config['agent_num'],
        pos_emb=False,
        value_sqrt=False,
        value_linear=True,
        add=True,
        norm=True,
        ff=True,
        dropout=0.05
    ).to(gpu)
    model_loss_record = train(tr_set, model, config, gpu)
    if model_loss_record[-1] < 1:
        break

labels, preds = test(tt_set, model, gpu)
#torch.save(model,'model/demo_dataset2_fold0.pth')

AUC = roc_auc_score(labels, preds)
precision, recall, _ = precision_recall_curve(labels, preds)
AUPR = auc(recall, precision)
preds = np.array([1 if p > 0.5 else 0 for p in preds])
ACC = accuracy_score(labels, preds)
P = precision_score(labels, preds)
R = recall_score(labels, preds)
F1 = f1_score(labels, preds)

print(AUC, AUPR, ACC, P, R, F1)
