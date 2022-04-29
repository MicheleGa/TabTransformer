import torch
import torch.nn.functional as F
from tab_transformer_pytorch import TabTransformer
from data_visualization import plot_roc_multiclass


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                print('\t\tINFO: Early stopping')
                self.early_stop = True


class Feedforward(torch.nn.Module):
    def __init__(self, input_size, num_classes, drop_out, m, n, num_input_columns=None):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        if num_input_columns is None:
            # case standard mlp model
            self.first_layer_hidden_size = m * input_size
            self.second_layer_hidden_size = n * input_size
        else:
            # case mlp transformer component
            self.first_layer_hidden_size = m * num_input_columns
            self.second_layer_hidden_size = n * num_input_columns
        self.num_classes = num_classes
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.first_layer_hidden_size),
            torch.nn.SELU(),
            torch.nn.Dropout(drop_out),
            torch.nn.Linear(self.first_layer_hidden_size, self.second_layer_hidden_size),
            torch.nn.SELU(),
            torch.nn.Linear(self.second_layer_hidden_size, self.num_classes)
        )

    def forward(self, x):
        return self.model(x)


def get_auc_score(targets, y_pred_proba, model_type):
    targets = targets.detach().cpu().numpy()
    y_pred_proba = y_pred_proba.detach().cpu().numpy()

    return plot_roc_multiclass(targets, y_pred_proba, model_type, None, None, is_training=True)


def train_mlp_model(model, criterion, optimizer, data_loader, device):

    """MLP training epoch"""

    model.train()
    train_loss = 0.0
    y_train_proba = []
    y_train = []
    for data, targets in data_loader:
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()

        # Forward pass
        y_pred = model(data)
        # Compute Loss
        loss = criterion(y_pred, targets)
        class_probs_batch = [F.softmax(el, dim=0) for el in y_pred]

        # Backward pass
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        y_train_proba.append(class_probs_batch)
        y_train.append(targets)

    return train_loss, torch.cat([torch.stack(batch) for batch in y_train_proba]), torch.cat(y_train)


def train_tab_model(model, criterion, optimizer, data_loader, device):

    """TabTransformer training epoch"""

    model.train()
    train_loss = 0.0
    y_train_proba = []
    y_train = []
    for cat_data, cont_data, targets in data_loader:
        cat_data, cont_data, targets = cat_data.to(device), cont_data.to(device), targets.to(device)
        optimizer.zero_grad()

        # Forward pass
        y_pred = model(cat_data, cont_data)

        # Compute Loss
        loss = criterion(y_pred, targets)
        class_probs_batch = [F.softmax(el, dim=0) for el in y_pred]

        # Backward pass
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        y_train_proba.append(class_probs_batch)
        y_train.append(targets)

    return train_loss, torch.cat([torch.stack(batch) for batch in y_train_proba]), torch.cat(y_train)


def valid_mlp_model(model, criterion, data_loader, device):

    """MLP validation"""

    model.eval()
    val_loss = 0.0
    y_pred_proba = []
    y_val = []
    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)

            y_pred = model(data.float())
            loss = criterion(y_pred, targets)
            class_probs_batch = [F.softmax(el, dim=0) for el in y_pred]

            val_loss += loss.item()
            y_pred_proba.append(class_probs_batch)
            y_val.append(targets)

        return val_loss, torch.cat([torch.stack(batch) for batch in y_pred_proba]), torch.cat(y_val)


def valid_tab_model(model, criterion, data_loader, device):

    """TabTransformer validation"""

    model.eval()
    val_loss = 0.0
    y_pred_proba = []
    y_val = []
    with torch.no_grad():
        for cat_data, cont_data, targets in data_loader:
            cat_data, cont_data, targets = cat_data.to(device), cont_data.to(device), targets.to(device)

            y_pred = model(cat_data, cont_data)
            loss = criterion(y_pred, targets)
            class_probs_batch = [F.softmax(el, dim=0) for el in y_pred]

            val_loss += loss.item()
            y_pred_proba.append(class_probs_batch)
            y_val.append(targets)

        return val_loss, torch.cat([torch.stack(batch) for batch in y_pred_proba]), torch.cat(y_val)


def get_torch_model(model_type, config, dataset):

    """Returns Pytorch MLP or TabTransformer model"""

    if model_type == 'MLP':
        model = Feedforward(dataset.X.shape[1],
                            dataset.num_classes,
                            config['dropout'],
                            config['first_layer'],
                            config['second_layer'])

    else:
        embedding_dim = 32
        model = TabTransformer(
            categories=dataset.categories,
            num_continuous=dataset.continuous_features,
            dim=embedding_dim,  # hidden embedding dimension
            dim_out=dataset.num_classes,  # number of output classes
            depth=2,  # transformer number of layers
            heads=4,  # number of attention heads
            attn_dropout=0.2,
            ff_dropout=0.2
        )

        # TabTransformer mlp components set as in the paper
        mlp_input_size = len(dataset.categories) * embedding_dim + dataset.continuous_features
        mlp = Feedforward(mlp_input_size,
                          dataset.num_classes,
                          config['dropout'],
                          config['first_layer'],
                          config['second_layer'],
                          len(dataset.categories) + dataset.continuous_features)

        model.mlp = mlp

    return model


def torch_model_training(history, model, model_type, config, optimizer, criterion, train_loader, val_loader,
                         early_stopping, writer=None):

    """
    Neural networks model training with early stopping and
    optional visualization with Tensorbaord
    """

    max_epochs = 0
    for epoch in range(config['max_nums_epochs']):
        if model_type == 'MLP':
            train_loss, y_train_proba, y_train_targets = train_mlp_model(model, criterion, optimizer, train_loader,
                                                                         config['device'])
            val_loss, y_val_proba, y_val_targets = valid_mlp_model(model, criterion, val_loader, config['device'])
        else:
            train_loss, y_train_proba, y_train_targets = train_tab_model(model, criterion, optimizer, train_loader,
                                                                         config['device'])
            val_loss, y_val_proba, y_val_targets = valid_tab_model(model, criterion, val_loader, config['device'])

        history['train_loss'].append(train_loss / len(train_loader.dataset))
        history['val_loss'].append(val_loss / len(val_loader.dataset))
        history['train_roc_auc'].append(get_auc_score(y_train_targets, y_train_proba, model_type))
        history['val_roc_auc'].append(get_auc_score(y_val_targets, y_val_proba, model_type))

        # writer.add_scalar('Loss/train', train_loss / len(train_loader.dataset), epoch)
        # writer.add_scalar('Loss/val', val_loss / len(val_loader.dataset), epoch)
        # writer.add_scalar('ROC AUC/train', get_auc_score(y_train_targets, y_train_proba, model_type))
        # writer.add_scalar('ROC AUC/val', get_auc_score(y_val_targets, y_val_proba, model_type))

        # count number of epochs
        max_epochs = epoch
        # early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            break
    # writer.flush()

    return history, max_epochs
