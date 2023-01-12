from model_functions import *
from models import *
import config


def train_model(hp):
    train_data = prepare_dataset(config.path_prefix + config.kgc_path + hp.city + '/train' + config.path_suffix)
    valid_data = prepare_dataset(config.path_prefix + config.kgc_path + hp.city + '/valid' + config.path_suffix)
    test_data = prepare_dataset(config.path_prefix + config.kgc_path + hp.city + '/test' + config.path_suffix)
    model = GTMiner(device=hp.device, finetuning=hp.finetuning, lm=hp.lm, n_relationships=config.n_relationships)
    model = model.to(hp.device)
    train(model, train_data, valid_data, test_data, config.save_path_kgc, hp)


def search_aois(hp):
    if hp.fe == 'bert':
        train_x_tensor, train_y_tensor = prepare_dataset_BertFE(config.path_prefix + config.classification_path +
                                                                hp.city + '/train' + config.path_suffix)
        valid_x_tensor, valid_y_tensor = prepare_dataset_BertFE(config.path_prefix + config.classification_path +
                                                                hp.city + '/valid' + config.path_suffix)
        test_x_tensor, test_y_tensor = prepare_dataset_BertFE(config.path_prefix + config.classification_path + hp.city
                                                              + '/test' + config.path_suffix)
        model = BertFE(device=hp.device, finetuning=True, lm=config.default_model)
        model = model.to(hp.device)
        train_BertFE(model, train_x_tensor, train_y_tensor, valid_x_tensor, valid_y_tensor, test_x_tensor, test_y_tensor
                     , config.save_path_classification, hp)

    elif hp.fe == 'lstm':
        glove_model = load_glove_model()
        train_x_tensor, train_y_tensor = prepare_dataset_LSTMFE(config.path_prefix + config.classification_path +
                                                                hp.city + '/train' + config.path_suffix, glove_model)
        valid_x_tensor, valid_y_tensor = prepare_dataset_LSTMFE(config.path_prefix + config.classification_path +
                                                                hp.city + '/valid' + config.path_suffix, glove_model)
        test_x_tensor, test_y_tensor = prepare_dataset_LSTMFE(config.path_prefix + config.classification_path + hp.city
                                                              + '/test' + config.path_suffix, glove_model)

        model = LSTMFE(device=hp.device, input_size=config.glove_size)
        model = model.to(hp.device).double()
        train_LSTMFE(model, train_x_tensor, train_y_tensor, valid_x_tensor, valid_y_tensor, test_x_tensor,
                     test_y_tensor, config.save_path_classification, hp)

    else:
        print('Error: Unknown Feature Extractor!')
        return
