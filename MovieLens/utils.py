from os import remove, listdir, sep
from os.path import join
import pickle
import numpy as np
import torch


def get_dataframe(csvs, df_name):
    for csv in csvs:
        if csv.endswith('ml-25m' + sep + df_name) or csv.endswith('imdb' + sep + df_name):
            return csv


def make_model_name(model_type, fold, config, model_name=''):
    """Utility function to make significant names for every model configuration"""

    if model_type == 'MLP':
        model_name = f'mlp{fold + 1}'
    elif model_type == 'TABT':
        model_name = f'tabt{fold + 1}'
    elif model_type == 'GBDT':
        model_name = f'gbdt{fold + 1}'
    elif model_type == 'LR':
        model_name = f'lr{fold + 1}'
    else:
        model_name = f'svm{fold + 1}'

    for key, value in config.items():
        if key != 'device':
            model_name += '_' + key + str(value)

    if model_type == 'MLP' or model_type == 'TABT':
        model_name += '.pt'
    else:
        model_name += '.pkl'

    return model_name


def output(config, recording_file, hpo_config=True):

    """Utility function to print model configuration"""

    string = ''
    for key, value in config.items():
        if hpo_config:
            string += f'\n\t\t\t{key} {value}'
        else:
            string += f'\n\t\t{key} {value}'

    if hpo_config:
        print(f'\t\tHyperparameters: ' + string)
    else:
        print(f'\tBest model: ' + string)
    with open(recording_file, 'a') as rec_file:
        if hpo_config:
            rec_file.write(f'\t\tHyperparameters: ' + string + '\n')
        else:
            rec_file.write(f'\tBest model: ' + string + '\n\n')


def compute_training_statistics(k, model_type, fold_perf_hpo):

    """Gather loss and ROC AUC score from training and validation oa a particular model configuration"""

    # calculate loss and final ROC AUC score of inner 5-fold cv
    mean_train_loss_f, mean_val_loss_f, mean_train_roc_auc_f, mean_val_roc_auc_f = [], [], [], []
    std_train_loss_f, std_val_loss_f, std_train_roc_auc_f, std_val_roc_auc_f = [], [], [], []

    # with MLP you gather metrics for every epoch, while other models
    # returns directly avg ROC AUC score for the inner cv fold
    # save normalizer/scaler for the model with highest roc_auc in hpo rounds

    for f in range(1, k + 1):
        if model_type == 'MLP' or model_type == 'TABT':
            mean_train_loss_f.append(np.mean(fold_perf_hpo[f'fold{f}'][0]['train_loss']))
            std_train_loss_f.append(np.std(fold_perf_hpo[f'fold{f}'][0]['train_loss']))

            mean_val_loss_f.append(np.mean(fold_perf_hpo[f'fold{f}'][0]['val_loss']))
            std_val_loss_f.append(np.std(fold_perf_hpo[f'fold{f}'][0]['val_loss']))

            mean_train_roc_auc_f.append(np.mean(fold_perf_hpo[f'fold{f}'][0]['train_roc_auc']))
            std_train_roc_auc_f.append(np.std(fold_perf_hpo[f'fold{f}'][0]['train_roc_auc']))

            mean_val_roc_auc_f.append(np.mean(fold_perf_hpo[f'fold{f}'][0]['val_roc_auc']))
            std_val_roc_auc_f.append(np.std(fold_perf_hpo[f'fold{f}'][0]['val_roc_auc']))
        else:
            mean_val_roc_auc_f.extend(fold_perf_hpo[f'fold{f}'][0]['val_roc_auc'])

    # mean_val_roc_auc_f store roc_auc for each hpo round
    best_hpo_round = np.argmax(mean_val_roc_auc_f)
    # normalizer = fold_perf_hpo[f'fold{best_hpo_round + 1}'][1]
    scaler = fold_perf_hpo[f'fold{best_hpo_round + 1}'][2]

    print(f'\t\t{model_type} performance of inner {k} fold cross validation:')
    if model_type == 'MLP' or model_type == 'TABT':
        print(
            '\t\tAverage Training Loss - {:.3f} \u00B1 {:.3f}'
            '\n\t\tAverage Val Loss - {:.3f} \u00B1 {:.3f}'
            '\n\t\tAverage Training ROC AUC score - {:.3f} \u00B1 {:.3f}'
            '\n\t\tAverage Val ROC AUC score - {:.3f} \u00B1 {:.3f}'.format(
                np.mean(mean_train_loss_f),
                np.mean(std_train_loss_f),
                np.mean(mean_val_loss_f),
                np.mean(std_val_loss_f),
                np.mean(mean_train_roc_auc_f),
                np.mean(std_train_roc_auc_f),
                np.mean(mean_val_roc_auc_f),
                np.mean(std_val_roc_auc_f)
            ))
    else:
        print('\t\tAverage Val ROC AUC score - {:.3f} \u00B1 {:.3f}'.format(
            np.mean(mean_val_roc_auc_f),
            np.std(mean_val_roc_auc_f)
        ))

    # if normalizer is used then change None to normalizer
    if model_type == 'MLP' or model_type == 'TABT':
        roc_auc = np.mean(mean_val_roc_auc_f) - np.mean(std_val_roc_auc_f)
        return roc_auc, mean_val_roc_auc_f, std_val_roc_auc_f, None, scaler
    else:
        roc_auc = np.mean(mean_val_roc_auc_f) - np.std(mean_val_roc_auc_f)
        return roc_auc, mean_val_roc_auc_f, None, None, scaler


def remove_models(model_name, model_type, models_path, hyperparams_history, outer_fold):

    """If a more perforing configuration is found then old model saved is deleted"""

    # remove previous model (and their normalizer/scaler) that performs worser
    for file in listdir(models_path):
        if not file.startswith('normalizer_') and not file.startswith('scaler_'):
            if file != model_name and file.startswith(model_type.lower() + str(outer_fold + 1)):
                if file.endswith('.pt'):
                    # remove mlp/tab normalizers and scalers
                    # remove(join(models_path, 'normalizer_' + file[:-3] + '.pkl'))
                    remove(join(models_path, 'scaler_' + file[:-3] + '.pkl'))
                else:
                    # remove sklearn model normalizers and scalers
                    # remove(join(models_path, 'normalizer_' + file[:-4] + '.pkl'))
                    remove(join(models_path, 'scaler_' + file[:-4] + '.pkl'))
                # remove models
                remove(join(models_path, file))

    # remove last model from hyperparams
    for key in hyperparams_history.keys():
        hyperparams_history[key] = hyperparams_history[key][:-1]

    return hyperparams_history


def save_model(models_path, model_type, model_name, model, config, hyperparams_history, normalizer, scaler,
               max_epochs=0):

    """Save best model and keep track of its configuration"""

    # every single model is saved only if it's ROC AUC score is better than the current best model
    for key, value in config.items():
        if key != 'device':
            hyperparams_history[key].append(value)

    # save model
    if model_type == 'MLP' or model_type == 'TABT':
        torch.save(model, join(models_path, model_name))
        # save normalizer and scaler
        # pickle.dump(normalizer, open(join(models_path, 'normalizer_' + model_name[:-3] + '.pkl'), 'wb'))
        pickle.dump(scaler, open(join(models_path, 'scaler_' + model_name[:-3] + '.pkl'), 'wb'))

    else:
        path = join(models_path, model_name)
        pickle.dump(model, open(path, 'wb'))
        # save normalizer and scaler
        # pickle.dump(normalizer, open(join(models_path, 'normalizer_' + model_name[:-4] + '.pkl'), 'wb'))
        pickle.dump(scaler, open(join(models_path, 'scaler_' + model_name[:-4] + '.pkl'), 'wb'))

    return hyperparams_history


def get_best_model(hyperparams_history):

    """Get best model with respect to ROC AUC score"""

    mean_roc_auc = hyperparams_history['roc_auc_mean'][0]
    std_roc_auc = hyperparams_history['roc_auc_std'][0]
    print('\t\tBest model ROC AUC score: {:.3f} \u00B1 {:.3f}'.format(mean_roc_auc, std_roc_auc))

    config = {}

    # gather best model params
    for key in hyperparams_history.keys():
        if key != 'roc_auc_mean' and key != 'roc_auc_std':
            config[key] = hyperparams_history[key][0]

    return config, mean_roc_auc, std_roc_auc
