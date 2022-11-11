import os
import time
import pickle
from os import environ, makedirs
from os.path import join
from pathlib import Path
from shutil import rmtree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from lightgbm import LGBMClassifier
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn import SELU, CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from tab_transformer_pytorch import TabTransformer

environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
torch.backends.cudnn.benchmark = False


class TabTorchDataset(Dataset):
    def __init__(self, cat_data, cont_data, labels):
        self.X_cat = torch.LongTensor(cat_data)
        self.X_cont = torch.FloatTensor(cont_data)
        self.y = torch.LongTensor(labels)

    def __len__(self):
        return self.X_cat.shape[0]

    def __getitem__(self, idx):
        return self.X_cat[idx, :], self.X_cont[idx, :], self.y[idx]


def tree_method(X, y, plots_folder):
    print('---- GBDT training ----')
    X_train, X_test, y_train, y_test = train_test_split(X.to_numpy(), y, test_size=0.33, stratify=y, random_state=42)

    # predict with LightGBMClassifier
    # GBDT hyperparameters
    config = {'random_state': 42,
              'objective': 'binary',
              'num_leaves': 16,
              'max_depth': 8,
              'min_child_samples': 7,
              'learning_rate': 0.1,
              'n_estimators': 10}

    model = LGBMClassifier(**config)
    model.fit(X_train, y_train)

    # predict probabilities
    gbdt_probs = model.predict_proba(X_test)

    # print classification report
    y_pred = np.argmax(gbdt_probs, axis=1)
    targets_name = [str(i) for i in np.arange(0, gbdt_probs.shape[1])]
    print(classification_report(y_test, y_pred, target_names=targets_name, zero_division=0))

    # keep probabilities for the positive outcome only
    gbdt_probs = gbdt_probs[:, 1]

    # generate a no skill prediction (majority class)
    ns_probs = [0 for _ in range(len(y_test))]

    # calculate scores
    ns_auc = roc_auc_score(y_test, ns_probs)
    gbdt_auc = roc_auc_score(y_test, gbdt_probs)

    # summarize scores
    print('No Skill: ROC AUC=%.3f' % ns_auc)
    print('GBDT: ROC AUC=%.3f' % gbdt_auc)

    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, gbdt_probs)

    # plot the roc curve for GBDT
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='GBDT')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(plots_folder + os.sep + 'gbdt_' + dataset_name[:-4] + '_roc_curve.jpg')
    plt.show()


def train_tab_model(model, criterion, optimizer, data_loader, device, writer):
    model.train()
    train_loss = 0.0
    for cat_data, cont_data, targets in data_loader:
        cat_data, cont_data, targets = cat_data.to(device), cont_data.to(device), targets.to(device)
        optimizer.zero_grad()

        # Forward pass
        y_pred = model(cat_data, cont_data)
        # Compute Loss
        loss = criterion(torch.squeeze(y_pred), targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    return train_loss


def tab_transformer(dataset_name, cat_data, cont_data, y, categories, continuous_features, plots_folder):
    print('---- TabTransformer training ----')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # TabTransformer hyperparameters
    config = {'random_state': 42,
              'device': device,
              'batch_sizes': 64,
              'max_nums_epochs': 10,
              'learning_rate': 1e-3,
              'weight_decay': 1e-5
              }

    # pytorch Dataset class
    train_idx, test_idx = train_test_split(np.arange(len(cat_data)), test_size=0.33, stratify=y, random_state=42)

    # standardize train data
    scaler = StandardScaler()
    scaler.fit(cont_data[train_idx, :])
    cont_data_train = scaler.transform(cont_data[train_idx, :])
    cont_data_test = scaler.transform(cont_data[test_idx, :])

    training_set = TabTorchDataset(cat_data[train_idx, :], cont_data_train, y[train_idx])
    test_set = TabTorchDataset(cat_data[test_idx, :], cont_data_test, y[test_idx])

    train_loader = DataLoader(training_set, batch_size=config['batch_sizes'], shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=config['batch_sizes'], shuffle=True, drop_last=True)

    # model definition
    model = TabTransformer(
        categories=categories,
        num_continuous=continuous_features,
        dim=32,  # hidden embedding dimension
        dim_out=2,  # number of output classes
        depth=6,  # transformer number of layers
        heads=8,  # number of attention heads
        attn_dropout=0.2,
        ff_dropout=0.2,
        mlp_hidden_mults=(4, 2),  # relative multiples of each hidden dimension of the last mlp to logits
        mlp_act=SELU()
    )

    model.to(config['device'])
    # BCEWithLogitLoss combines a sigmoid layer with a BinaryCrossEntropy loss
    criterion = CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config['learning_rate'],
                                  weight_decay=config['weight_decay'])

    # TabTransformer training
    tb_dir = Path('runs')
    if not tb_dir.exists():
        makedirs(tb_dir)
    writer = SummaryWriter(log_dir=str(tb_dir) + os.sep + dataset_name[:-4])

    history = []
    for epoch in range(config['max_nums_epochs']):
        train_loss = train_tab_model(model, criterion, optimizer, train_loader, config['device'], writer)

        history.append(train_loss / len(train_loader.dataset))

        print(f'Epoch {epoch}: loss {train_loss / len(train_loader.dataset)}')

        writer.add_scalar('Loss/train', train_loss / len(train_loader.dataset), epoch)
    writer.flush()

    # save model for embedding visualization
    torch.save(model, 'model/' + dataset_name[:-4] + '_tab_transformer.pt')

    # test TabTransformer
    model.eval()
    with torch.no_grad():
        y_pred = []
        y_test = []
        for cat_data, cont_data, targets in test_loader:
            cat_data, cont_data, targets = cat_data.to(device), cont_data.to(device), targets.to(device)

            output = model(cat_data, cont_data)
            class_probs_batch = [F.softmax(el, dim=0) for el in output]

            y_pred.append(class_probs_batch)
            y_test.append(targets)
        y_pred_proba = torch.cat([torch.stack(batch) for batch in y_pred])
        y_test = torch.cat(y_test)
        y_pred = y_pred_proba.argmax(dim=1, keepdim=True)

        # detach data with torch.no_grad() and send them back to cpu
        y_pred_proba = y_pred_proba.cpu().numpy()
        y_test = y_test.cpu().numpy()
        y_pred = y_pred.cpu().numpy()

        targets_name = [str(i) for i in np.arange(0, y_pred_proba.shape[1])]
        print(classification_report(y_test, y_pred, target_names=targets_name, zero_division=0))

        # keep probabilities for the positive outcome only
        tabt_probs = y_pred_proba[:, 1]

        # generate a no skill prediction (majority class)
        ns_probs = [0 for _ in range(len(y_test))]

        # calculate scores
        ns_auc = roc_auc_score(y_test, ns_probs)
        tabt_auc = roc_auc_score(y_test, tabt_probs)

        # summarize scores
        print('No Skill: ROC AUC=%.3f' % ns_auc)
        print('TABT: ROC AUC=%.3f' % tabt_auc)

        # calculate roc curves
        ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
        lr_fpr, lr_tpr, _ = roc_curve(y_test, tabt_probs)

        # plot the roc curve for GBDT
        plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
        plt.plot(lr_fpr, lr_tpr, marker='.', label='TABT')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.savefig(plots_folder + os.sep + 'tabt_' + dataset_name[:-4] + '_roc_curve.jpg')
        plt.show()


def preprocessing(dataset_name, folder):
    if dataset_name == 'income_evaluation.zip':
        df = pd.read_csv('data/' + dataset_name)
    else:
        df = pd.read_csv('data/' + dataset_name, sep=';')

    # print some infos of df
    print(dataset_name + ' dataframe: ', df.shape)
    print(df.head())

    # no null values, good
    print('Null values count:\n', df.isnull().sum())

    # look out, columns name starts with a blank space (for 1995_income)
    print('Columns:\n', df.columns)
    if dataset_name == 'income_evaluation.zip':
        for col in df.columns:
            if col.startswith(' '):
                df.rename({col: col[1:]}, axis=1, inplace=True)

    # categorical/continuous values distinction
    print('Columns data type:\n', df.dtypes)
    target = df.iloc[:, -1]
    target = LabelEncoder().fit_transform(target)
    if dataset_name == 'income_evaluation.zip':
        df.drop('income', axis=1, inplace=True)
    else:
        df.drop('y', axis=1, inplace=True)

    # target class is a bit unbalanced
    positive_class = df.value_counts(target)
    print('Target variable value counts:\n', positive_class)

    # encode categorical data and standardize continuous features
    categorical_data = df.select_dtypes(include=['object'])
    num_categorical_data = categorical_data.shape[1]
    print('Number of categorical features:', num_categorical_data)
    num_continuous_features = df.shape[1] - num_categorical_data
    print('Number of continuous features:', num_continuous_features)
    print('Number of unique values per categorical feature:\n', categorical_data.nunique())
    num_classes_per_category = categorical_data.nunique().to_numpy().tolist()

    i = 0  # identifies position of categorical feature in df.columns
    for col in categorical_data.columns:
        print(col, ':\t', categorical_data[col].unique())
        data = np.array(categorical_data[col])
        le = LabelEncoder()
        data = le.fit_transform(data)

        # save label encoder for embedding visualization
        if folder is not None:
            pickle.dump(le, open(str(folder) + os.sep + f'encoder_{col}{i}.pkl', 'wb'))

        categorical_data[col] = data
        df[col] = data
        i += 1

    continuous_cols = []
    for col in df.columns:
        if col not in categorical_data.columns:
            continuous_cols.append(col)
    continuous_data = df[continuous_cols]

    data_set = {
        'df': df,
        'target': target
    }

    tabt_data_set = {
        'cat_data': categorical_data,
        'cont_data': continuous_data,
        'num_classes_cat': num_classes_per_category,
        'num_cont_feat': num_continuous_features
    }

    return data_set, tabt_data_set


if __name__ == '__main__':
    start = time.time()
    # datasets = ['income_evaluation.zip', 'bank-full.zip']
    datasets = ['income_evaluation.zip']

    # make empty directories to save embeddings, plots and TabTransformer model
    embeddings_folder = Path('embedding_visualization')
    if Path(embeddings_folder).exists():
        rmtree(embeddings_folder)
    makedirs(embeddings_folder)

    plots_folder = Path('plots')
    if Path(plots_folder).exists():
        rmtree(plots_folder)
    makedirs(plots_folder)

    model_folder = Path('model')
    if Path(model_folder).exists():
        rmtree(model_folder)
    makedirs(model_folder)

    # make an empty directory to save label encoders
    encoder_folder = Path(join('embedding_visualization', 'label_encoders'))
    if Path(encoder_folder).exists():
        rmtree(encoder_folder)
    makedirs(encoder_folder)

    for dataset_name in datasets:
        print('**** ' + dataset_name + '****')
        dataset_encoder_folder = Path(join('embedding_visualization', 'label_encoders', dataset_name[:-4]))
        if not Path(dataset_encoder_folder).exists():
            makedirs(dataset_encoder_folder)

        # clean NaN, encode categorical data and standardize data
        dataset, tabt_dataset = preprocessing(dataset_name, dataset_encoder_folder)

        # train GBDT since its insensible to data scale
        tree_method(dataset['df'], dataset['target'], str(plots_folder))

        # train TabTransformer
        tab_transformer(dataset_name,
                        tabt_dataset['cat_data'].to_numpy(),
                        tabt_dataset['cont_data'].to_numpy(),
                        dataset['target'],
                        tabt_dataset['num_classes_cat'],
                        tabt_dataset['num_cont_feat'],
                        str(plots_folder))

    print(f'Total execution time: {time.time() - start} s')
