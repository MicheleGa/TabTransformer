from os import sep
import itertools
import numpy as np
from sklearn.preprocessing import Normalizer, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import TorchDataset, TabTorchDataset
from performance_measure import performance_measure
from torch_models import EarlyStopping, get_torch_model, torch_model_training
from utils import *
from data_preprocessing import normalize_scale_and_balance
from data_visualization import plot_roc_multiclass


def get_indices_for_hpo(k, model_type, dataset, train_idx):
    splits_hpo = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    if model_type != 'TABT':
        X_train = dataset.X[train_idx, :]
        y_train = dataset.y[train_idx]

        indices_hpo = splits_hpo.split(X_train, y_train)

        return indices_hpo, X_train, None, None, y_train
    else:
        X_train_cat = dataset.X_cat[train_idx, :]
        X_train_cont = dataset.X_cont[train_idx, :]
        y_train = dataset.y[train_idx]

        indices_hpo = splits_hpo.split(X_train_cat, y_train)

        return indices_hpo, None, X_train_cat, X_train_cont, y_train


def inner_5_fold_cv(k, train_idx, hyperparams_history, dataset, config, models_path, max_roc_auc, outer_fold,
                    recording_file):

    """Performs inner k-fold cross validation to select best configuration for a model"""

    model_type = models_path.split(sep)[-1]

    indices_hpo, X_train, X_train_cat, X_train_cont, y_train = get_indices_for_hpo(k, model_type, dataset, train_idx)

    model = None
    fold_perf_hpo = {}
    max_epochs = 0  # for MLP and TABT

    # inner cross validation of a particular model
    for fold_hpo, (train_idx_hpo, val_idx) in enumerate(indices_hpo):

        print(f'\t\tHPO Fold {fold_hpo + 1}')
        with open(recording_file, 'a') as rec_file:
            rec_file.write(f'\t\tHPO Fold {fold_hpo + 1}\n')

        # normalize (L2 norm) and scale (min-max scaling) training set and apply norm/scale to validation set
        # normalizer = Normalizer()
        scaler = MinMaxScaler()

        if model_type != 'TABT':
            X, X_val, y, y_val = normalize_scale_and_balance(model_type,
                                                             y_train,
                                                             train_idx_hpo,
                                                             val_idx,
                                                             None,  # normalizer
                                                             scaler,
                                                             X_train=X_train)

        else:
            X, X_val, y, y_val = normalize_scale_and_balance(model_type,
                                                             y_train,
                                                             train_idx_hpo,
                                                             val_idx,
                                                             None,  # normalizer
                                                             scaler,
                                                             X_train_cat=X_train_cat,
                                                             X_train_cont=X_train_cont
                                                             )

        if model_type == 'MLP' or model_type == 'TABT':
            # writer = SummaryWriter(log_dir=join(models_path, f'runs{sep}hpo_fold{fold_hpo + 1}'))

            # Build a pytorch dataset to feed model with dataframe
            if model_type == 'MLP':
                training_set = TorchDataset(X, y)
                val_set = TorchDataset(X_val, y_val)
            else:
                X_cat, X_cont = X[:, :3], X[:, 3:]
                X_cat_val, X_cont_val = X_val[:, :3], X_val[:, 3:]

                training_set = TabTorchDataset(X_cat, X_cont, y)
                val_set = TabTorchDataset(X_cat_val, X_cont_val, y_val)

            train_loader = DataLoader(training_set, batch_size=config['batch_sizes'], shuffle=True, drop_last=True)
            val_loader = DataLoader(val_set, batch_size=config['batch_sizes'], shuffle=True, drop_last=True)

            # init early stopping
            early_stopping = EarlyStopping()

            model = get_torch_model(model_type, config, dataset)

            model.to(config['device'])
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(model.parameters(),
                                          lr=config['learning_rate'],
                                          weight_decay=config['weight_decay'])

            # data_iter = iter(train_loader)
            # if model_type == 'MLP':
            #     data, _ = data_iter.next()
            #     data = data.to(config['device'])
            #     writer.add_graph(model, data)

        else:
            if model_type == 'GBDT':
                model = LGBMClassifier(**config)
            elif model_type == 'SVM':
                model = SVC(**config)
            else:
                model = LogisticRegression(**config)

        history = {'train_loss': [], 'val_loss': [], 'train_roc_auc': [], 'val_roc_auc': []}

        if model_type == 'MLP' or model_type == 'TABT':

            history, num_epochs = torch_model_training(history,
                                                       model,
                                                       model_type,
                                                       config,
                                                       optimizer,
                                                       criterion,
                                                       train_loader,
                                                       val_loader,
                                                       early_stopping)
                                                       # writer)
            max_epochs = num_epochs

        else:
            # train sklearn model
            model.fit(X, y)

            # mesaure performance using ROC AUC score as metric (since dataset is unbalanced)
            if model_type != 'SVM':
                y_pred_proba = model.predict_proba(X_val)
            else:
                p = np.array(model.decision_function(X_val))  # decision is a voting function
                y_pred_proba = np.exp(p) / np.sum(np.exp(p), axis=1, keepdims=True)  # softmax after the voting

            history['val_roc_auc'].append(plot_roc_multiclass(y_val, y_pred_proba, model_type, None, None,
                                                              is_training=True))

        fold_perf_hpo[f'fold{fold_hpo + 1}'] = (history, None, scaler)

    roc_auc, mean_val_roc_auc_f, std_val_roc_auc_f, normalizer, scaler = compute_training_statistics(k, model_type,
                                                                                                     fold_perf_hpo)

    if roc_auc > max_roc_auc:
        # new best model

        # adjust number of epochs for torch models
        if model_type == 'MLP' or model_type == 'TABT':
            config['max_nums_epochs'] = max_epochs + 1

        model_name = make_model_name(model_type, outer_fold, config)
        hyperparams_history = remove_models(model_name, model_type, models_path, hyperparams_history, outer_fold)

        # save configuration in order to get best model later in outer k-fold
        hyperparams_history['roc_auc_mean'].append(np.mean(mean_val_roc_auc_f))

        if model_type == 'MLP' or model_type == 'TABT':
            hyperparams_history['roc_auc_std'].append(np.mean(std_val_roc_auc_f))
        else:
            hyperparams_history['roc_auc_std'].append(np.std(mean_val_roc_auc_f))

        with open(recording_file, 'a') as rec_file:
            rec_file.write(f'\t\t{model_type} performance of inner {k} fold cross validation:\n')
            rec_file.write('\t\tAverage Val ROC AUC score - {:.3f} '.format(
                np.mean(mean_val_roc_auc_f)
            ))
            if model_type == 'MLP' or model_type == 'TABT':
                rec_file.write('\u00B1 {:.3f}\n\n'.format(
                    np.mean(std_val_roc_auc_f)
                ))
            else:
                rec_file.write('\u00B1 {:.3f}\n\n'.format(
                    np.std(mean_val_roc_auc_f)
                ))

        if max_epochs != 0:
            hyperparams_history = save_model(models_path,
                                             model_type,
                                             model_name,
                                             model,
                                             config,
                                             hyperparams_history,
                                             None,
                                             scaler,
                                             max_epochs)
        else:
            hyperparams_history = save_model(models_path,
                                             model_type,
                                             model_name,
                                             model,
                                             config,
                                             hyperparams_history,
                                             None,
                                             scaler)

    return hyperparams_history, roc_auc


def grid_search_cv_loop(k, models_path, model_type, hyperparams, hyperparams_history, dataset, train_idx, fold,
                        recording_file, device=None):

    """
    Iterate through hyperparameters search space.
    At the end, parameters of the model with maximum ROC AUC score on validation
    are returned
    """

    max_roc_auc = 0.0
    if model_type == 'MLP' or model_type == 'TABT':
        for max_num_epochs, batch, lr, wd, drop, fl, sl in hyperparams:
            config = {'max_nums_epochs': max_num_epochs,
                      'batch_sizes': batch,
                      'learning_rate': lr,
                      'weight_decay': wd,
                      'dropout': drop,
                      'first_layer': fl,
                      'second_layer': sl,
                      'device': device}

            output(config, recording_file)

            hyperparams_history, roc_auc = inner_5_fold_cv(k,
                                                           train_idx,
                                                           hyperparams_history,
                                                           dataset,
                                                           config,
                                                           models_path,
                                                           max_roc_auc,
                                                           fold,
                                                           recording_file)
            if roc_auc > max_roc_auc:
                max_roc_auc = roc_auc

    elif model_type == 'GBDT':

        for rs, obj, max_dep, num_leaves, min_child_samples, learning_rate, n_estimators in hyperparams:
            config = {'random_state': rs,
                      'objective': obj,
                      'max_depth': max_dep,
                      'num_leaves': num_leaves,
                      'min_child_samples': min_child_samples,
                      'learning_rate': learning_rate,
                      'n_estimators': n_estimators}

            output(config, recording_file)

            hyperparams_history, roc_auc = inner_5_fold_cv(k,
                                                           train_idx,
                                                           hyperparams_history,
                                                           dataset,
                                                           config,
                                                           models_path,
                                                           max_roc_auc,
                                                           fold,
                                                           recording_file)
            if roc_auc > max_roc_auc:
                max_roc_auc = roc_auc

    elif model_type == 'SVM':

        for rs, c, tol, kernel, gamma, max_it in hyperparams:
            config = {'random_state': rs,
                      'C': c,
                      'tol': tol,
                      'kernel': kernel,
                      'gamma': gamma,
                      'max_iter': max_it}

            output(config, recording_file)

            hyperparams_history, roc_auc = inner_5_fold_cv(k,
                                                           train_idx,
                                                           hyperparams_history,
                                                           dataset,
                                                           config,
                                                           models_path,
                                                           max_roc_auc,
                                                           fold,
                                                           recording_file)
            if roc_auc > max_roc_auc:
                max_roc_auc = roc_auc

    elif model_type == 'LR':

        for rs, pen, c, tol, solv, mc, max_it in hyperparams:
            config = {'random_state': rs,
                      'penalty': pen,
                      'C': c,
                      'tol': tol,
                      'solver': solv,
                      'multi_class': mc,
                      'max_iter': max_it}

            output(config, recording_file)

            hyperparams_history, roc_auc = inner_5_fold_cv(k,
                                                           train_idx,
                                                           hyperparams_history,
                                                           dataset,
                                                           config,
                                                           models_path,
                                                           max_roc_auc,
                                                           fold,
                                                           recording_file)
            if roc_auc > max_roc_auc:
                max_roc_auc = roc_auc

    return hyperparams_history


def define_model_hyperparams(model_type):

    """Returns a dictionary containing the possible configurations for a given model"""

    space = {}
    if model_type == 'MLP' or model_type == 'TABT':
        # Feedforward and TabTransformer hyperparams
        space['max_nums_epochs'] = [100]
        space['batch_sizes'] = [32, 64, 128]
        space['learning_rate'] = [1e-1, 1e-2, 1e-3]
        space['weight_decay'] = [1e-1, 1e-3, 1e-5]
        space['dropout'] = [0.1, 0.2, 0.3]
        space['first_layer'] = [2, 4, 8]  # first layer hidden size is m*l, l=input size
        space['second_layer'] = [2, 3]  # second layer hidden size is n*l, l=input size

    elif model_type == 'GBDT':
        # LightGBMClassifier
        space['random_state'] = [42]
        space['objective'] = ['multiclass']
        space['max_depth'] = [2, 4, 8]
        space['num_leaves'] = [4, 8, 16, 32, 50]
        space['min_child_samples'] = [2, 5, 7, 10, 20, 50]
        space['learning_rate'] = [0.1, 0.01, 0.001]
        space['n_estimators'] = [5, 10, 50, 500]

    elif model_type == 'SVM':
        # SVC
        space['random_state'] = [42]
        space['C'] = [0.01, 1, 10, 100]
        space['tol'] = [10, 1, 0.1, 0.01]
        space['kernel'] = ['linear', 'rbf']
        space['gamma'] = [0.001, 0.1, 1]
        space['max_iter'] = [2500]

    else:
        # LogisticRegression
        space['random_state'] = [42]
        space['penalty'] = ['l2']
        space['C'] = [0.001, 0.01, 1, 10, 100]
        space['tol'] = [0.01, 0.1, 0.01, 0.001]
        space['solver'] = ['saga', 'lbfgs']
        space['multi_class'] = ['multinomial']
        space['max_iter'] = [2500]

    return space


def hpo(fold, k, models_path, dataset, train_idx, device, recording_file):

    """
    Hyperparameters optimization step: returns best parameters and training
    results for a model after doing an inner k-fold cross validation
    """

    model_type = models_path.split(sep)[-1]

    space = define_model_hyperparams(model_type)

    hyperparams = itertools.product(*space.values())

    # store roc_auc score and config of each model to select best configuration
    hyperparams_history = {'roc_auc_mean': [],
                           'roc_auc_std': []}

    for key in space.keys():
        hyperparams_history[key] = []

    # grid search cv
    print('\t\tGrid search cv ...')
    with open(recording_file, 'a') as rec_file:
        rec_file.write('\t\tGrid search cv:\n')

    if model_type == 'MLP' or model_type == 'TABT':
        hyperparams_history = grid_search_cv_loop(k,
                                                  models_path,
                                                  model_type,
                                                  hyperparams,
                                                  hyperparams_history,
                                                  dataset,
                                                  train_idx,
                                                  fold,
                                                  recording_file,
                                                  device)
    else:
        hyperparams_history = grid_search_cv_loop(k,
                                                  models_path,
                                                  model_type,
                                                  hyperparams,
                                                  hyperparams_history,
                                                  dataset,
                                                  train_idx,
                                                  fold,
                                                  recording_file,
                                                  device)

    # now hyperparams_history store the config of the best model
    return hyperparams_history


def model_train_and_test(dataset, k, models_path, recording_file, plots_folder):

    """Model definition and training using nested k-fold cross validation"""

    model_type = models_path.split(sep)[-1]
    device = None

    if model_type == 'MLP' or model_type == 'TABT':
        print(f'Training and testing {model_type} with Pytorch ...')
        # data ready, now test if cuda is available to use gpu
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Device: {device}')
        with open(recording_file, 'a') as rec_file:
            rec_file.write(f'----{model_type} train and test on {device}----\n\n')

    else:
        print(f'Training and testing {model_type} model ...')
        with open(recording_file, 'a') as rec_file:
            rec_file.write(f'----{model_type} train and test----\n\n')

    print('5-fold cross-validation ...')
    # 5-fold as in TabTransformer paper
    splits = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    if model_type != 'TABT':
        indices = splits.split(dataset.X, dataset.y)
    else:
        indices = splits.split(dataset.X_cat, dataset.y)

    fold_perf = {}

    for fold, (train_idx, test_idx) in enumerate(indices):
        print(f'\tFold {fold + 1}')
        with open(recording_file, 'a') as rec_file:
            rec_file.write(f'\tFold {fold + 1}\n')

        # grid search for hyperparameters optimization
        hyperparams_history = hpo(fold, k, models_path, dataset, train_idx, device, recording_file)

        config, mean_roc_auc, std_roc_auc = get_best_model(hyperparams_history)
        model_name = make_model_name(model_type, fold, config)

        auc_score = performance_measure(models_path,
                                        model_name,
                                        model_type,
                                        config,
                                        dataset,
                                        test_idx,
                                        fold,
                                        recording_file,
                                        plots_folder,
                                        device)

        output(config, recording_file, hpo_config=False)

        print('\tFold {} ROC AUC score: {:.3f}'.format(fold + 1, auc_score))

        with open(recording_file, 'a') as rec_file:
            rec_file.write('\tFold {} ROC AUC score: {:.3f}\n\n'.format(fold + 1, auc_score))

        # store model roc_auc on training (mean/std), and auc score on test set
        fold_perf[f'fold{fold + 1}'] = (mean_roc_auc, std_roc_auc, auc_score)

    # summarize results stored in fold_perf of outer 5-fold cv
    train_roc_auc_mean = []
    train_roc_auc_std = []
    test_auc_score = []
    for k in range(1, k + 1):
        train_roc_auc_mean.append(fold_perf[f'fold{k}'][0])
        train_roc_auc_std.append(fold_perf[f'fold{k}'][1])
        test_auc_score.append(fold_perf[f'fold{k}'][2])

    train_roc_auc_mean, train_roc_auc_std = np.mean(train_roc_auc_mean), np.mean(train_roc_auc_std)
    mean_test_auc_score, std_test_auc_score = np.mean(test_auc_score), np.std(test_auc_score)

    print('Final {} model roc_auc: {:.3f} \u00B1 {:.3f}'.format(model_type, train_roc_auc_mean, train_roc_auc_std))
    print('Final {} model mean AUC score : {:.3f} \u00B1 {:.3f}'.format(model_type,
                                                                        mean_test_auc_score,
                                                                        std_test_auc_score))

    with open(recording_file, 'a') as rec_file:
        rec_file.write('Final {} model roc_auc: {:.3f} \u00B1 {:.3f}\n'.format(
            model_type, train_roc_auc_mean, train_roc_auc_std))
        rec_file.write('Final {} model mean AUC score : {:.3f} \u00B1 {:.3f}\n\n'.format(model_type,
                                                                                         mean_test_auc_score,
                                                                                         std_test_auc_score))

    return train_roc_auc_mean, train_roc_auc_std, mean_test_auc_score, std_test_auc_score
