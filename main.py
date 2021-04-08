import numpy as np
from utils import *

class Classifier(Forward, Backward):

    def __init__(self, layers):
        super(Classifier, self).__init__()

        self.layer_dims = layers
        self.train_losses = []
        self.train_accuracy = []
        self.val_losses = []
        self.val_accuracy = []
        self.parameters = Forward().initialize_parameters(layer_dims=self.layer_dims)

    def train(self, X_train, y_train, X_val, y_val, epoches, learning_rate):

        for i in range(0, epoches+1):

            # Training
            out, caches = Forward().model_forward(X_train, self.parameters)
            train_loss = Forward().compute_loss(out, y_train)
            train_acc = Forward().compute_accuracy(out, y_train)
            grads =  Backward().model_backward(out, y_train, caches)            
            self.parameters = Backward().update_parameters(self.parameters, grads, learning_rate)

            # Validation
            out, caches = Forward().model_forward(X_val, self.parameters)
            val_loss = Forward().compute_loss(out, y_val)
            val_acc = Forward().compute_accuracy(out, y_train)

            if i% 50 == 0:

                print("Epoch {} =>  Training Loss : {t_loss:.2f};   Training Acc : {t_acc:.2f}%   Validation Loss : {v_loss:.2f};   Validation Acc : {v_acc:.2f}%;   ".format(i, t_loss=train_loss, t_acc=train_acc, v_loss=val_loss, v_acc=val_acc,))

                self.train_losses.append(train_loss)
                self.train_accuracy.append(train_acc)
                self.val_losses.append(val_loss)
                self.val_accuracy.append(val_acc)

        return self.train_losses, self.train_accuracy, self.val_losses, self.val_accuracy

    def predict(self, X, y):

        m = X.shape[1]
        n = len(self.parameters) // 2 # number of layers in the neural network
        p = np.zeros((1,m))

        # Forward() propagation
        prediction_array, caches = Forward().model_forward(X, self.parameters)

        acc = Forward().compute_accuracy(prediction_array, y)

        return prediction_array, acc


def main():

    X_train = np.load("dataset/X_train.npy")
    y_train = np.load("dataset/y_train.npy")

    X_test = np.load("dataset/X_test.npy")
    y_test = np.load("dataset/y_test.npy")

    classes = ['Non-Cat', 'Cat']
    layer_dims = [12288, 20, 7, 5, 1]
    epoches = 1000
    learning_rate = 0.01

    clf = Classifier(layer_dims)
    vis = Visualize()

    vis.data_visualize(X_train, y_train)

    X_train = X_train.reshape(X_train.shape[0], -1).T

    X_train = X_train/255.

    X_test_ = X_test.reshape(X_test.shape[0], -1).T

    X_test_ = X_test_/255.

    train_loss, train_acc, val_loss, val_acc = clf.train(X_train, y_train, X_test_, y_test, epoches, learning_rate)

    vis.plot(train_loss, val_loss, 'Loss', ["Training Loss", "Validation Loss"], 'Loss vs Epoches')
    vis.plot(train_acc, val_acc, 'Accuracy', ["Training Accuracy", "Validation Accuracy"], 'Accuracy vs Epoches')

    predictions_array, acc = clf.predict(X_test_, y_test)

    vis.test_plot(X_test, y_test, predictions_array, classes)

    print("\n\nAccuracy of Classifier on Test Dataset: {acc: .2f}%".format(acc = acc*100))

main()