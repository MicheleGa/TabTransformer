from os.path import join
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from data_visualization import plot_roc_multiclass
from dataset import TorchDataset, TabTorchDataset


def test_mlp_model(model, data_loader, model_type, fold, recording_file, plots_folder, device):

    """MLP test"""

    model.eval()
    with torch.no_grad():
        y_pred = []
        y_test = []
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)

            output = model(data.float())
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
        with open(recording_file, 'a') as rec_file:
            rec_file.write(classification_report(y_test, y_pred, target_names=targets_name, zero_division=0) + '\n\n')

        return plot_roc_multiclass(y_test, y_pred_proba, model_type, fold, plots_folder)


def test_tab_model(model, data_loader, model_type, fold, recording_file, plots_folder, device):

    """TabTransformer test"""

    model.eval()
    with torch.no_grad():
        y_pred = []
        y_test = []
        for cat_data, cont_data, targets in data_loader:
            cat_data, cont_data, targets = cat_data.to(device), cont_data.to(device), targets.to(device)

            output = model(cat_data, cont_data.float())
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
        with open(recording_file, 'a') as rec_file:
            rec_file.write(classification_report(y_test, y_pred, target_names=targets_name, zero_division=0) + '\n\n')

        return plot_roc_multiclass(y_test, y_pred_proba, model_type, fold, plots_folder)


def test_sklearn_model(model, model_type, X_test, y_test, fold, recording_file, plots_dir):

    """GBDT, SVM and LR test"""

    # SVC output should be treated different in order to get probabilities
    if model_type != 'SVM':
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
    else:
        p = np.array(model.decision_function(X_test))  # decision is a voting function
        y_pred_proba = np.exp(p) / np.sum(np.exp(p), axis=1, keepdims=True)  # softmax after the voting
        y_pred = np.argmax(y_pred_proba, axis=1)

    targets_name = [str(i) for i in range(0, y_pred_proba.shape[1])]

    print(classification_report(y_test, y_pred, target_names=targets_name, zero_division=0))
    with open(recording_file, 'a') as rec_file:
        rec_file.write(classification_report(y_test, y_pred, target_names=targets_name, zero_division=0) + '\n\n')

    return plot_roc_multiclass(y_test, y_pred_proba, model_type, fold, plots_dir)


def performance_measure(models_path, model_name, model_type, config, dataset, test_idx, fold, recording_file,
                        plots_folder, device):

    """
    Performance measure stage.
    Load best model from grid search and its min-max scaler in order
    to apply it to the test set
    """

    if model_type == 'MLP' or model_type == 'TABT':
        model = torch.load(join(models_path, model_name))
        print(model)
        # restore normalizer/scaler
        # normalizer = pickle.load(open(join(models_path, 'normalizer_' + model_name[:-3] + '.pkl'), 'rb'))
        scaler = pickle.load(open(join(models_path, 'scaler_' + model_name[:-3] + '.pkl'), 'rb'))

        if model_type == 'MLP':
            # apply normalizer/scaler on test data
            # X_test = scaler.transform(normalizer.transform(dataset.X[test_idx, :]))
            X_test = scaler.transform(dataset.X[test_idx, :])
            y_test = dataset.y[test_idx]
            test_set = TorchDataset(X_test, y_test)

        else:
            X_test_cat, X_test_cont = dataset.X_cat[test_idx, :], dataset.X_cont[test_idx, :]
            # apply normalizer on test data
            # X_test_cont = scaler.transform(normalizer.transform(X_test_cont))
            X_test_cont = scaler.transform(X_test_cont)
            y_test = dataset.y[test_idx]
            test_set = TabTorchDataset(X_test_cat, X_test_cont, y_test)

        test_loader = DataLoader(test_set, batch_size=config['batch_sizes'], shuffle=True, drop_last=True)

        if model_type == 'MLP':
            auc_score = test_mlp_model(model, test_loader, model_type, fold, recording_file, plots_folder, device)
        else:
            auc_score = test_tab_model(model, test_loader, model_type, fold, recording_file, plots_folder, device)

    else:
        model = pickle.load(open(join(models_path, model_name), 'rb'))
        # restore normalizer
        # normalizer = pickle.load(open(join(models_path, 'normalizer_' + model_name[:-4] + '.pkl'), 'rb'))
        scaler = pickle.load(open(join(models_path, 'scaler_' + model_name[:-4] + '.pkl'), 'rb'))
        # apply normalizer on test data
        # X_test = normalizer.transform(dataset.X[test_idx, :])
        # X_test = scaler.transform(normalizer.transform(dataset.X[test_idx, :]))
        X_test = scaler.transform(dataset.X[test_idx, :])
        # X_test = dataset.X[test_idx, :]
        y_test = dataset.y[test_idx]

        auc_score = test_sklearn_model(model, model_type, X_test, y_test, fold, recording_file, plots_folder)

    return auc_score
