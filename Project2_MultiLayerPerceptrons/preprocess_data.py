'''preprocess_data.py
Preprocessing data in STL-10 image dataset
Duilio Lucio, Vivian Hu
CS343: Neural Networks
Project 2: Multilayer Perceptrons
'''
import numpy as np
import load_stl10_dataset


def preprocess_stl(imgs, labels):
    '''Preprocesses stl image data for training by a MLP neural network

    Parameters:
    ----------
    imgs: unint8 ndarray  [0, 255]. shape=(Num imgs, height, width, RGB color chans)

    Returns:
    ----------
    imgs: float64 ndarray [0, 1]. shape=(Num imgs N,)
    Labels: int ndarray. shape=(Num imgs N,). Contains int-coded class values 0,1,...,9

    TODO:
    1) Cast imgs to float64
    2) Flatten height, width, color chan dims. New shape will be (num imgs, height*width*chans)
    3) Treating the pixels as features, standardize the features "seperately"
    4) Fix class labeling. Should span 0, 1, ..., 9 NOT 1,2,...10
    '''
    # Cast imgs to float64 and scale to [0, 1]
    imgs = imgs.astype(np.float64) / 255.0
    # Flatten per image: N, W, H, C -> (N, H*W*C)
    num_imgs = imgs.shape[0]
    imgs = imgs.reshape(num_imgs, -1)
    # Standardize features "seperately"
    mean = imgs.mean(axis = 0)
    std = imgs.std(axis = 0)
    eps = 1e-8 # avoids divide by zero
    imgs = (imgs - mean) / (std + eps)
    # Class labeling fix, 0...9 instead of 1 ... 10
    labels = labels.astype(int) - 1
    
    return imgs, labels

def create_splits(data, y, n_train_samps=3500, n_test_samps=500, n_valid_samps=500, n_dev_samps=500):
    '''Divides the dataset up into train/test/validation/development "splits" (disjoint partitions)

    Parameters:
    ----------
    data: float64 ndarray. Image data. shape=(Num imgs, height*width*chans)
    y: ndarray. int-coded labels.

    Returns:
    ----------
    None if error
    x_train (training samples),
    y_train (training labels),
    x_test (test samples),
    y_test (test labels),
    x_val (validation samples),
    y_val (validation labels),
    x_dev (development samples),
    y_dev (development labels)

    TODO:
    1) Divvy up the images into train/test/validation/development non-overlapping subsets (see return vars)

    NOTE: Resist the urge to shuffle the data here! It is best to shuffle the data "live"
    during training so hold off on shuffling your data here.
    '''
    if n_train_samps + n_test_samps + n_valid_samps + n_dev_samps != len(data):
        samps = n_train_samps + n_test_samps + n_valid_samps + n_dev_samps
        print(f'Error! Num samples {samps} does not equal num images {len(data)}!')
        return
    
    # align y w/ data length
    if len(y) != len(data):
        print(f'Error! Num labels {len(y)} does not equal num images {len(data)}')
        return
    # Boundaries slicing
    i0 = 0
    i1 = i0 + n_train_samps
    i2 = i1 + n_test_samps
    i3 = i2 + n_valid_samps
    i4 = i3 + n_dev_samps # needs to equal len(data)
    
    # Divvy up images into train/test/val/dev 
    x_train, y_train = data[i0:i1], y[i0:i1]
    x_test, y_test = data[i1:i2], y[i1:i2]
    x_val, y_val = data[i2:i3], y[i2:i3]
    x_dev, y_dev = data[i3:i4], y[i3:i4]
    
    return x_train, y_train, x_test, y_test, x_val, y_val, x_dev, y_dev


def load_stl10(n_train_samps=3500, n_test_samps=500, n_valid_samps=500, n_dev_samps=500):
    '''Automates the process of:
    - loading in the STL-10 dataset and labels
    - preprocessing
    - creating the train/test/validation/dev splits.

    Returns:
    ----------
    None if error
    x_train (training samples),
    y_train (training labels),
    x_test (test samples),
    y_test (test labels),
    x_val (validation samples),
    y_val (validation labels),
    x_dev (development samples),
    y_dev (development labels)
    '''
    # Load STL-10 
    try: 
        imgs, labels = load_stl10_dataset.load_stl10_dataset()
    except AttributeError:
        imgs, labels = load_stl10_dataset.load() # fallback, load() is used sometimes
    # Preprocess
    data, y = preprocess_stl(imgs, labels)
    # Validate Splits
    total = n_train_samps + n_test_samps + n_valid_samps + n_dev_samps
    if total != len(data):
        print(f'Error: Requested split total {total} does not equal dataset size {len(data)}')
    # Create Splits
    return create_splits(data, y, 
                         n_train_samps=n_train_samps, 
                         n_test_samps=n_test_samps,
                         n_valid_samps=n_valid_samps,
                         n_dev_samps=n_dev_samps)