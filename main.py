import numpy as np
from utils import *

class Classifier(Forward, Backward):

    def __init__(self, layers):
        super(Classifier, self).__init__()

        self.layer_dims = layers
        self.losses = []
        self.accuracies = []
        self.parameters = Forward().initialize_parameters(layer_dims=self.layer_dims)
        
    def train(self, X, y, epoches, learning_rate):

        for i in range(0, epoches+1):

            out, caches = Forward().model_forward(X, self.parameters)
            loss = Forward().compute_loss(out, y)
            _, acc = Forward().compute_accuracy(out, y)
            grads =  Backward().model_backward(out, y, caches)            
            self.parameters = Backward().update_parameters(self.parameters, grads, learning_rate)

            if i% 50 == 0:

                print("Epoch {} =>  loss : {loss:.2f};   Accuracy : {acc:.2f}%;   ".format(i, loss=loss, acc=acc, ))

                self.losses.append(loss)
                self.accuracies.append(acc)
        
        return self.losses, self.accuracies
                
    def predict(self, X, y):
        
        m = X.shape[1]
        n = len(self.parameters) // 2 # number of layers in the neural network
        p = np.zeros((1,m))
        
        # Forward() propagation
        prediction_array, caches = Forward().model_forward(X, self.parameters)

        p, acc = Forward().compute_accuracy(prediction_array, y)
            
        return prediction_array, p, acc


def main():

    X_train = np.load("dataset/X_train.npy")
    y_train = np.load("dataset/y_train.npy")

    classes = ['Cat', 'Non Cat']
    layer_dims = [12288, 20, 7, 5, 1]
    epoches= 1000
    learning_rate = 0.0075

    clf = Classifier(layer_dims)
    """vis = Visualize()

    vis.data_visualize(X_train, y_train)"""

    X_train = X_train.reshape(X_train.shape[0], -1).T

    X_train = X_train/255.

    loss, acc = clf.train(X_train, y_train, epoches, learning_rate)

    """vis.plot(loss, 'Loss', 'Loss vs Epoches')
    vis.plot(acc, 'Accuracy', 'Accuracy vs Epoches')"""

    X_test = np.load("dataset/X_test.npy")
    y_test = np.load("dataset/y_test.npy")

    X_test_ = X_test.reshape(X_test.shape[0], -1).T

    X_test_ = X_test_/255.

    prediction_array, y_pred, acc = clf.predict(X_test_, y_test)

    #vis.test_plot(X_test, y_test, y_pred, prediction_array, classes)

    print("\n\nAccuracy of Classifier on Test Dataset: {acc: .2f}%".format(acc = acc*100))

main()