# Created 5 Jan 2024
# Modified 5 Jan 2024

import warnings
warnings.filterwarnings('ignore')

import numpy as np

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# Keras
import keras
from keras.models import load_model
from keras.losses import categorical_crossentropy
from keras.optimizers.legacy import Adam

# ART
from art.utils import load_dataset, get_file
from art.estimators.classification import KerasClassifier
from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod, ProjectedGradientDescent
from art.defences.trainer import AdversarialTrainer
from art import config

###########################################################################
#
class CARTAttacks:
    def __init__(self):
        print('CConsole object is created')

    def CreateModel(self, strPath, x_train, y_train, bArt):
        cnnModel = tf.keras.Sequential()
        cnnModel.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3),
                strides=1, activation='relu',
                input_shape=(28, 28, 1),))
        cnnModel.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        cnnModel.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
                strides=1, activation='relu',
                input_shape=(23, 23, 4),))
        cnnModel.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        cnnModel.add(keras.layers.Flatten())
        cnnModel.add(keras.layers.Dense(128, activation='relu'))
        cnnModel.add(keras.layers.Dense(10, activation='softmax'))

        cnnModel.compile(loss=categorical_crossentropy,
            optimizer=Adam(learning_rate=1e-4),
            metrics=['accuracy'])

        if bArt:
            classifier = KerasClassifier(clip_values=(min_, max_),
                                        model=cnnModel,
                                        use_logits=False)
            classifier.fit(x_train, y_train, nb_epochs=10,
                        batch_size=128, validation_split=0.1, verbose=1)
            classifier.model.save(strPath)
        else:
            aBatchSz = 128
            aEpochs = 10
            hist = cnnModel.fit(x_train, y_train, 
                    batch_size=aBatchSz, 
                    epochs=aEpochs, 
                    validation_split=0.1, 
                    verbose=1)
            cnnModel.save(strPath)
        print('Model is created successfully')

    def CreateRobustModel(self, strPath):
        cnnRobustModel = keras.Sequential()
        cnnRobustModel.add(
            keras.layers.Conv2D(
                filters=32,
                kernel_size=(3, 3),
                strides=1,
                activation='relu',
                input_shape=(28, 28, 1),
            )
        )
        cnnRobustModel.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        cnnRobustModel.add(
            keras.layers.Conv2D(
                filters=64,
                kernel_size=(3, 3),
                strides=1,
                activation='relu',
                input_shape=(23, 23, 4),
            )
        )
        cnnRobustModel.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        cnnRobustModel.add(keras.layers.Flatten())
        # Note: the robust classifier has the same architecture as above,
        # except the first dense layer has **1024** instead of **128** units.
        # (This was recommend by Madry et al. (2017),
        # *Towards Deep Learning Models Resistant to Adversarial Attacks*)
        cnnRobustModel.add(keras.layers.Dense(1024, activation='relu'))
        cnnRobustModel.add(keras.layers.Dense(10, activation='softmax'))

        cnnRobustModel.compile(
            loss=categorical_crossentropy,
            optimizer=Adam(learning_rate=1e-4),
            metrics=['accuracy'],
        )

        classifier = KerasClassifier(clip_values=(min_, max_), 
                                     model=cnnRobustModel, 
                                     use_logits=False)

        classifier.fit(x_train, y_train,
            nb_epochs=10, batch_size=128,
            validation_split=0.1, verbose=1)

        classifier.model.save(strPath)
        print('Model is created successfully')

    def InitModel(self, strModalPath, min_, max_):
        mLoadedModel = load_model(strModalPath)
        mLoadedModel.summary()
        print('Model name: ', strModalPath, 
              'Min: ', min_, 'Max: ', max_)
        nn_model = KerasClassifier(clip_values=(min_, max_),
                                   model=mLoadedModel,
                                   use_logits=False)
        return nn_model

    def TestModel(self, strPrompt, nn_model, x_test, y_test):
        print('Testing Data', x_test.shape)

        x_test_pred = np.argmax(nn_model.predict(x_test), axis=1)
        nb_correct_pred = np.sum(x_test_pred == np.argmax(y_test, axis=1))

        print(strPrompt)
        print('Correctly classified: ', nb_correct_pred)
        print('Incorrectly classified: ', len(x_test) - nb_correct_pred)

###################################################################

def NormalNetwork(strCmd):
    obj = CARTAttacks()

    if strCmd == 'LOAD':
        strModelName = './models/mnist_cnn_original.h5'
        # Original test data:
        # Correctly classified:  9842
        # Incorrectly classified:  158
        # Testing Data (10000, 28, 28, 1)
        # Adversarial test data:
        # Correctly classified:  31
        # Incorrectly classified:  9969

        # strModelName = './models/mnist_cnn_local.h5'
        # Original test data:
        # Correctly classified:  9833
        # Incorrectly classified:  167
        # Testing Data (10000, 28, 28, 1)
        # Adversarial test data:
        # Correctly classified:  43
        # Incorrectly classified:  9957
    elif strCmd == 'DOWNLOAD':
        # load model from C:\Users\3058388\.art\data\mnist_cnn_original.h5
        strModelName = get_file(
            'mnist_cnn_original.h5',
            extract=False,
            path=config.ART_DATA_PATH,
            url='https://www.dropbox.com/s/p2nyzne9chcerid/mnist_cnn_original.h5?dl=1',
        )
        print('Model is downloaded: ', strModelName)
    elif strCmd == 'CREATE':
        strModelName = './models/mnist_cnn_local.h5'
        obj.CreateModel(strModelName, x_train, y_train, True)

    nn_model = obj.InitModel(strModelName, min_, max_)
    obj.TestModel('Original test data:', nn_model, x_test, y_test)

    oAttack = FastGradientMethod(nn_model, eps=0.5)
    x_test_adv = oAttack.generate(x_test, y_test)
    obj.TestModel('Adversarial test data:', nn_model, x_test_adv, y_test)

def RobustNetwork(strCmd):
    obj = CARTAttacks()

    if strCmd == 'LOAD':
        strModelName = './models/mnist_cnn_robust.h5'
    elif strCmd == 'DOWNLOAD':
        strModelName = get_file('mnist_cnn_robust.h5',
                                extract=False,
                                path=config.ART_DATA_PATH,
                                url='https://www.dropbox.com/s/yutsncaniiy5uy8/mnist_cnn_robust.h5?dl=1')

    nn_model = obj.InitModel(strModelName, min_, max_)
    obj.TestModel('Robust Model with original test data:',
                  nn_model, x_test, y_test)

    oAttack = FastGradientMethod(nn_model, eps=0.5)
    x_test_adv_robust = oAttack.generate(x_test, y_test)
    obj.TestModel('Robust Model with adversarial test data:',
                  nn_model, x_test_adv_robust, y_test)


###################################################################
if __name__ == '__main__':
    print('Adversarial Training MNIST')

    (x_train, y_train), (x_test, y_test), min_, max_ = load_dataset('mnist')

    NormalNetwork('LOAD')
    #NormalNetwork('CREATE')

    RobustNetwork('LOAD')
