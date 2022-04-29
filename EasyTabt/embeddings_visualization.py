import pickle
from shutil import rmtree
import pandas as pd
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from pathlib import Path
from tensorboardX import SummaryWriter
from easy_tabt import preprocessing, TabTorchDataset


class DeepFeatures(torch.nn.Module):

    def __init__(self,
                 model,
                 batch_size,
                 experiment_name):

        super(DeepFeatures, self).__init__()

        self.model = model
        self.batch_size = batch_size
        self.model.eval()
        self.name = experiment_name
        self.writer = None

    def generate_contextual_embeddings(self, x_categ):
        return self.model(x_categ)

    def generate_embeddings(self, x_categ):
        return self.model.embeds(x_categ)

    def write_embeddings(self, x_categ):
        assert len(os.listdir(input_data_folder)) == 0, "Images folder must be empty"
        assert len(os.listdir(embs_folder)) == 0, "Embeddings folder must be empty"

        # generate embeddings
        embs = self.generate_embeddings(x_categ)
        # generate contextual embeddings
        contextual_embs = self.generate_contextual_embeddings(x_categ)

        # in order to get one dimensional embedding, concatenate embs
        # that has size (BATCH_SIZE, NUMBER_OF_CATEGORICAL_FEATURES, EMB_LENGTH),
        # i.e. (256, 8, 32); in this way final embedding will be (256, 32 * 8)

        # detach from graph
        x_categ = x_categ.detach().cpu().numpy()
        embs = torch.reshape(embs, (self.batch_size, 32 * x_categ.shape[1]))
        embs = embs.detach().cpu().numpy()
        contextual_embs = torch.reshape(contextual_embs, (self.batch_size, 32 * x_categ.shape[1]))
        contextual_embs = contextual_embs.detach().cpu().numpy()

        # start writing to output folders
        for i in range(len(contextual_embs)):
            key = str(np.random.random())[-7:]
            np.save(str(input_data_folder) + '/' + key + '.npy', x_categ[i])
            np.save(str(embs_folder) + '/' + key + '.npy', embs[i])
            np.save(str(contextual_embs_folder) + '/' + key + '.npy', contextual_embs[i])

        # save metadata as label for each categorical feature embedding
        # feature[-1] identifies position in x_categ
        for feature, encoder in label_encoders.items():
            column = pd.DataFrame({feature: x_categ[:, int(feature[-1])]})
            feature_labels = encoder.inverse_transform(column[feature].to_numpy())
            np.save(os.path.join(data_folder, feature + '_metadata.npy'), feature_labels)

        return True

    def _create_writer(self):
        if self.name is None:
            name = 'Experiment_' + str(np.random.random())
        else:
            name = self.name

        dir_name = os.path.join(tb_folder, name)

        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        else:
            print("Warning: logfile already exists")
            print("logging directory: " + str(dir_name))

        logdir = dir_name
        self.writer = SummaryWriter(logdir=logdir)

    def create_tensorboard_log(self, x_categ):

        if self.writer is None:
            self._create_writer()

        # add graph
        self.writer.add_graph(self.model, x_categ)

        # read in
        all_embeddings = [np.load(os.path.join(embs_folder, p))
                          for p in os.listdir(embs_folder)
                          if p.endswith('.npy')]
        all_contextual_embeddings = [np.load(os.path.join(contextual_embs_folder, p))
                                     for p in os.listdir(contextual_embs_folder)
                                     if p.endswith('.npy')]

        # stack into tensors
        all_embeddings = torch.Tensor(all_embeddings)
        all_contextual_embeddings = torch.Tensor(all_contextual_embeddings)

        metadatas = np.zeros(shape=(self.batch_size,))
        for feature in label_encoders.keys():
            metadata = np.load(os.path.join(data_folder, feature + '_metadata.npy'), allow_pickle=True)
            metadatas = np.column_stack((metadatas, metadata))

        metadatas = np.delete(metadatas, 0, axis=1)

        self.writer.add_embedding(all_embeddings, metadata=metadatas, tag='no_context')
        self.writer.add_embedding(all_contextual_embeddings, metadata=metadatas, tag='context')


def main():
    dataset, tabt_data_set = preprocessing(dataset_name, folder=None)

    # to generate embeddings we only care about categorical data
    data = TabTorchDataset(tabt_data_set['cat_data'].to_numpy(),
                           tabt_data_set['cont_data'].to_numpy(),
                           dataset['target'])

    batch_size = 256
    data_loader = DataLoader(data,
                             batch_size=batch_size,
                             shuffle=True)

    models_dir = Path('model')
    path = os.path.join(models_dir, dataset_name[:-4] + '_tab_transformer.pt')

    # load trained model and retrieve only transformer part
    model = torch.load(path)
    model = model.transformer
    model.to(device)
    print(model)
    deep_features = DeepFeatures(model=model,
                                 batch_size=batch_size,
                                 experiment_name='TABT')

    cat_data, _, _ = next(iter(data_loader))
    deep_features.write_embeddings(x_categ=cat_data.to(device))
    deep_features.create_tensorboard_log(x_categ=cat_data.to(device))


if __name__ == '__main__':
    data_folder = './embedding_visualization/data/Dataset'
    data_folder = Path(data_folder)
    if data_folder.exists():
        rmtree(data_folder)
    os.makedirs(data_folder)

    input_data_folder = './embedding_visualization/outputs/Dataset/Input_data'
    input_data_folder = Path(input_data_folder)
    if input_data_folder.exists():
        rmtree(input_data_folder)
    os.makedirs(input_data_folder)

    embs_folder = './embedding_visualization/outputs/Dataset/Embeddings'
    embs_folder = Path(embs_folder)
    if embs_folder.exists():
        rmtree(embs_folder)
    os.makedirs(embs_folder)

    contextual_embs_folder = './embedding_visualization/outputs/Dataset/ContextualEmbeddings'
    contextual_embs_folder = Path(contextual_embs_folder)
    if contextual_embs_folder.exists():
        rmtree(contextual_embs_folder)
    os.makedirs(contextual_embs_folder)

    tb_folder = './embedding_visualization/outputs/Tensorboard'
    tb_folder = Path(tb_folder)
    if tb_folder.exists():
        rmtree(tb_folder)
    os.makedirs(tb_folder)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # gather label encoders for inverse transform in order to label embeddings
    dataset_name = 'bank-full.zip'
    label_encoders = {}
    path = os.path.join('embedding_visualization', 'label_encoders', dataset_name[:-4])
    for file in os.listdir(path):
        if file.endswith('.pkl'):
            label_encoders[file[8:-4]] = pickle.load(open(path + '/' + file, 'rb'))

    main()
