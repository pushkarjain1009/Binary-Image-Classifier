import numpy as np
import matplotlib.pyplot as plt

class Activation_Function:
    def sigmoid(self, Z):

        A = 1/(1+np.exp(-Z))
        cache = Z

        return A, cache

    def relu(self, Z):

        A = np.maximum(0,Z)

        assert(A.shape == Z.shape)

        cache = Z 
        return A, cache


    def relu_backward(self, dA, cache):

        Z = cache
        dZ = np.array(dA, copy=True) # just converting dz to a correct object.

        dZ[Z <= 0] = 0

        assert (dZ.shape == Z.shape)

        return dZ

    def sigmoid_backward(self, dA, cache):

        Z = cache

        s = 1/(1+np.exp(-Z))
        dZ = dA * s * (1-s)

        assert (dZ.shape == Z.shape)

        return dZ


######################################################################################################################

class Forward(Activation_Function):

    def initialize_parameters(self, layer_dims):

        np.random.seed(1)

        parameters = {}

        L = len(layer_dims)            # number of layers in the network

        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) 
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

            assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
            assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))


        return parameters

    def linear_forward(self, A, W, b):

        Z = W.dot(A) + b

        assert(Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b)

        return Z, cache

    def linear_activation_forward(self, A_prev, W, b, activation):
        
        if activation == "sigmoid":

            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = Activation_Function().sigmoid(Z)

        elif activation == "relu":

            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = Activation_Function().relu(Z)
        
        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)

        return A, cache

    def model_forward(self, X, parameters):

        caches = []
        A = X
        L = len(parameters) // 2                  # number of layers in the neural network

        for l in range(1, L):
            A_prev = A 
            A, cache = self.linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")
            caches.append(cache)

        AL, cache = self.linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "sigmoid")
        caches.append(cache)

        assert(AL.shape == (1,X.shape[1]))

        return AL, caches

    def compute_loss(self, AL, Y):

        m = Y.shape[0]
        loss = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
        loss = np.squeeze(loss)      
        assert(loss.shape == ())

        return loss

    def compute_accuracy(self, probas, y):

        m = y.shape[0]
        p = np.zeros((1,m))

        for i in range(0, probas.shape[1]):
            if probas[0,i] > 0.5:
                p[0,i] = 1
            else:
                p[0,i] = 0


        return np.sum((p == y)/m)

        
################################################################################################################

class Backward(Activation_Function):

    def linear_backward(self, dZ, cache):

        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = 1./m * np.dot(dZ,A_prev.T)
        db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
        dA_prev = np.dot(W.T,dZ)

        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)

        return dA_prev, dW, db

    def linear_activation_backward(self, dA, cache, activation):
        linear_cache, activation_cache = cache

        if activation == "relu":
            dZ = super().relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        elif activation == "sigmoid":
            dZ = super().sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        return dA_prev, dW, db

    def model_backward(self, AL, Y, caches):

        grads = {}
        L = len(caches) 
        m = AL.shape[1]
        Y = Y.reshape(AL.shape) 

        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        current_cache = caches[L-1]
        grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(dAL, current_cache, activation = "sigmoid")

        for l in reversed(range(L-1)):

            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation = "relu")
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads

    def update_parameters(self, parameters, grads, learning_rate):

        L = len(parameters) // 2
        for l in range(L):
            #print(learning_rate * grads["dW" + str(l+1)])
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]

        return parameters


###################################################################################################################

class Visualize:

    def data_visualize(self, X, y):

        fig = plt.figure()
        for i in range(9):

            plt.subplot(3,3,i+1)
            plt.imshow(X[i])
            plt.title("Ground Truth: {}".format(y[i]))
        plt.show()

    def plot(self, arr1, arr2, y_label, legend, title):

        x = len(arr1)

        plt.plot([i*50 for i in range(x)], arr1)
        plt.plot([i*50 for i in range(x)], arr2)

        plt.xlabel('Epoch')
        plt.ylabel(y_label)
        plt.legend(legend)
        plt.title(title)
        plt.show()

    def plot_image(self, i, predictions_array, true_label, img, classes):

        prediction_array, true_label, img = predictions_array[0][i], true_label[i], img[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        plt.imshow(img, cmap = plt.cm.binary)

        if prediction_array >= 0.5:
            predicted_label = 1
        else:
            predicted_label = 0

        if predicted_label == true_label:
            color = 'green'
        else:
            color = 'red'

        plt.xlabel("{} {:2.0f}% ({})".format(classes[predicted_label],
                                            100*np.max([prediction_array, 1-prediction_array]),
                                            classes[true_label]),
                                            color=color)
        return

    def plot_value_array(self, i, predictions_array, true_label):

        prediction_array, true_label = predictions_array[0][i], true_label[i]

        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        thisplot = plt.bar(range(2), [1-prediction_array, prediction_array], color="#777777")
        plt.ylim([0,1])

        if prediction_array >= 0.5:
            predicted_label = 1
        else:
            predicted_label = 0

        thisplot[predicted_label].set_color('red')
        thisplot[true_label].set_color('green')

        return

    def test_plot(self, X_test, y_test, predictions_array, classes):

        num_rows = 3
        num_cols = 3
        num_images = num_rows*num_cols

        plt.figure(figsize=(8*num_cols, 4*num_rows))

        for i in range(num_images):

            plt.subplot(num_rows, 2*num_cols, 2*i+1)
            self.plot_image(i, predictions_array, y_test, X_test, classes)
            plt.subplot(num_rows, 2*num_cols, 2*i+2)
            self.plot_value_array(i, predictions_array, y_test)
            plt.xticks(range(2), classes)

        plt.show()