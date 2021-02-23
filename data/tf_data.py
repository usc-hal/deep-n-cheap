# =============================================================================
# Loading data using tf.keras
# Ziping Chen, USC
# =============================================================================

import tensorflow as tf
# import tensorflow_datasets as tfds # tf dataloader
import numpy as np
import os

# =============================================================================
# Data processing
# =============================================================================


def get_data(data_folder='./', dataset="mnist", val_split=1 / 5, augment=True):
    '''
    Args:
        dataset (string, Dataset from tf.keras.datasets): Currently supports MNIST, FMNIST, CIFAR10, CIFAR100
        val_split (float, optional): What fraction of training data to use for validation
            If not 0, val data is taken from end of training set. Eg: For val_split=1/5, last 10k images out of 50k for CIFAR are taken as val
            If 0, train set is complete train set (including val). Test data is returned as val set
            Defaults to 1/5
        augment (bool, optional): If True, do transformations
            Defaults to True

    Returns:
        dict: Keys 'train', 'val', 'test'
            Each is a dataset object which can be fed to a data loader
            If val_split is 0, test data is returned as 'val'

    '''
    # TODO: support data augmentation
    if dataset == 'mnist':
        (xtr, ytr), (xte, yte) = tf.keras.datasets.mnist.load_data()
        xtr = xtr.reshape((xtr.shape[0], -1))
        xte = xte.reshape((xte.shape[0], -1))
    elif dataset == 'fmnist':
        (xtr, ytr), (xte, yte) = tf.keras.datasets.fashion_mnist.load_data()
        xtr = xtr.reshape((xtr.shape[0], -1))
        xte = xte.reshape((xte.shape[0], -1))
    elif dataset == 'cifar10':
        (xtr, ytr), (xte, yte) = tf.keras.datasets.cifar10.load_data()
    elif dataset == 'cifar100':
        (xtr, ytr), (xte, yte) = tf.keras.datasets.cifar100.load_data()
    else:
        raise Exception("dataset not supported!!!")

    if abs(val_split) < 1e-8:
        # val_spilt is 0.0
        return xtr, ytr, xte, yte, xte, yte
    else:
        split = int((1 - val_split) * len(xtr))
        xva = xtr[split:]
        yva = ytr[split:]
        xtr = xtr[:split]
        ytr = ytr[:split]
        return xtr, ytr, xva, yva, xte, yte


def get_data_npz(data_folder='./', dataset='fmnist.npz', val_split=1 / 5, problem_type='classification'):
    '''
    Args:
        data_folder : Location of dataset
        dataset (string): <dataset name>.npz, must have 4 keys -- xtr, ytr, xte, yte
            xtr: (num_trainval_samples, num_features...)
            ytr: (num_trainval_samples,)
            xte: (num_test_samples, num_features...)
            yte: (num_test_samples,)
        val_split (float, optional): What fraction of training data to use for validation
            If not 0, val data is taken from end of training set. Eg: For val_split=1/6, last 10k images out of 60k for MNIST are taken as val
            If 0, train set is complete train set (including val). Test data is returned as val set
            Defaults to 1/5
        problem_type (float, reserved): Task/problem type. Reserved because torch_data.py needs this info.

    Returns:
        xtr (numpy array): Shape: (num_train_samples, num_features...)
        ytr (numpy array): Shape: (num_train_samples,)
        xva (numpy array): Shape: (num_val_samples, num_features...). This is xte if val_split = 0
        yva (numpy array): Shape: (num_val_samples,). This is yte if val_split = 0
        xte (numpy array): Shape: (num_test_samples, num_features...)
        yte (numpy array): Shape: (num_test_samples,)

    '''
    loaded = np.load(os.path.join(data_folder, dataset))
    xtr = loaded['xtr']
    ytr = loaded['ytr']
    xte = loaded['xte']
    yte = loaded['yte']

    if abs(val_split) < 1e-8:
        # val_spilt is 0.0
        return xtr, ytr, xte, yte, xte, yte
    else:
        split = int((1 - val_split) * len(xtr))
        xva = xtr[split:]
        yva = ytr[split:]
        xtr = xtr[:split]
        ytr = ytr[:split]
        return xtr, ytr, xva, yva, xte, yte
