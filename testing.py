import os
import numpy as np
import skimage
from skimage import io

def load_training_set():  
    arrow_train = os.path.join(os.getcwd(), 'training_sets','new_split_training2','arrows_train')
    notArrow_train = os.path.join(os.getcwd(), 'training_sets','new_split_training2','not_arrows_train')
    list_arrows = sorted(os.listdir(arrow_train))[2:]
    list_nots = sorted(os.listdir(notArrow_train))[2:]

    training_set = []
    training_labels = []

    for i, arrow in enumerate(list_arrows):
        if arrow != '.ipynb_checkpoints':
            try:
                image = io.imread(os.path.join(arrow_train, arrow))
            except:
                print(arrow)
                continue
            if type(image) == type(None) or image.shape != (500, 500):
                print(arrow)
            training_set.append(image)
            training_labels.append(1)

    for j, nots in enumerate(list_nots):
        if arrow != '.ipynb_checkpoints':
            try:
                image = io.imread(os.path.join(notArrow_train, nots))
            except:
                print(nots)
                continue
            if type(image) == type(None):
                print('NoneType: ' + arrow)
            elif image.shape != (500, 500):
                print(nots + str(image.shape))
            training_set.append(image)
            training_labels.append(0)

    training_set = np.array(training_set)
    training_set = training_set / 255.0
    training_set = training_set.reshape(len(training_set), 500, 500, 1)
    training_labels = np.array(training_labels)
    print(training_set.shape)
    print(training_labels.shape)
    return training_set, training_labels

def load_testing_set():
    # Load the directories with the arrows and not arrows
    arrow_test = os.path.join(os.getcwd(), 'training_sets','new_split_training2','arrows_test')
    notArrow_test= os.path.join(os.getcwd(), 'training_sets','new_split_training2','not_arrows_test')
    test_arrows = sorted(os.listdir(arrow_test))[1:]
    test_nots = sorted(os.listdir(notArrow_test))[1:]

    testing_set = []
    testing_labels = []
    # Load arrows to testing set
    for i, arrow in enumerate(test_arrows):
        if arrow != '.ipynb_checkpoints':
            image = io.imread(os.path.join(arrow_test, arrow))
            if type(image) == type(None) or image.shape != (500, 500):
                print(arrow)
            testing_set.append(image)
            testing_labels.append(1)
    # Load others to testing set
    for j, nots in enumerate(test_nots):
        if arrow != '.ipynb_checkpoints':
            image = io.imread(os.path.join(notArrow_test, nots))
            if type(image) == type(None):
                print('NoneType: ' + arrow)
            elif image.shape != (500, 500):
                print(nots + str(image.shape))
            testing_set.append(image)
            testing_labels.append(0)
    # convert to np.array and process (shape and reduce)
    testing_set = np.array(testing_set)
    testing_set = testing_set / 255.0
    testing_set = testing_set.reshape(len(testing_set), 500, 500, 1)
    testing_labels = np.array(testing_labels)
    print(testing_set.shape)
    print(testing_labels.shape)
    return testing_set, testing_labels