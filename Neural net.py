import tensorflow as tf
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import time

class NeuralNetwork:
    def __init__(self):
        self.conv_filters = np.random.rand(num_filters, filter_size, filter_size) * 0.1
        self.layer_weights = np.random.rand(10, flat_size) * 0.1
        self.layer_biases = np.random.rand(10)

    def forward(self, image):
        conv_output = np.zeros((num_filters, output_height, output_width))
        
        area = sliding_window_view(image, (filter_size, filter_size))
        area_reshaped = area.reshape(-1, filter_size * filter_size)

        filters_reshaped = self.conv_filters.reshape(num_filters, -1)

        conv_output = filters_reshaped @ area_reshaped.T
        conv_output = conv_output.reshape(num_filters, output_height, output_width)

        relu_output = np.maximum(conv_output, 0)

        output_flat = relu_output.flatten()

        logits = np.dot(self.layer_weights, output_flat) + self.layer_biases

        exp_layer = np.exp(logits - np.max(logits))
        probs = exp_layer / np.sum(exp_layer)

        return conv_output, relu_output, output_flat, logits, probs

    def backward(self, image, label, conv_output, relu_output, output_flat, probs):
        d_logits = probs.copy()
        d_logits[label] -= 1

        d_layer_weights = np.outer(d_logits, output_flat)
        d_layer_bias = d_logits

        d_flattened = np.dot(self.layer_weights.T, d_logits)
        d_relu_output = d_flattened.reshape(relu_output.shape)

        d_conv_output = d_relu_output * (conv_output > 0)

        d_conv_filter = np.zeros((num_filters, filter_size, filter_size))
        
        image_areas = sliding_window_view(image, (filter_size, filter_size))
        d_conv_filter = np.einsum('fij,ijxy->fxy', d_conv_output, image_areas)

        self.layer_weights -= learning_rate * d_layer_weights
        self.layer_biases -= learning_rate * d_layer_bias
        self.conv_filters -= learning_rate * d_conv_filter

        epsilon = 1e-9
        loss = -np.log(probs[label] + epsilon)

        return loss

    def train(self, images, labels, epochs):
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            for i in range(images.shape[0]):
                image = images[i]
                label = labels[i]

                conv_output, relu_output, flat, logits, probs = self.forward(image)

                loss = self.backward(image, label, conv_output, relu_output, flat, probs)
                total_loss += loss

                if np.argmax(probs) == label:
                    correct += 1
                
                if (i + 1) % 1000 == 0:
                    print(f"Image {i+1}, Avg Loss: {total_loss/(i+1):.4f}, Accuracy: {correct/(i+1):.4f}")

            print(f"Epoch {epoch+1} completed. Avg Loss: {total_loss/len(images):.4f}, Accuracy: {correct/len(images):.4f}")

        np.savez('cnn_weights.npz', conv_filters = self.conv_filters, layer_weights = self.layer_weights, 
                 layer_biases = self.layer_biases)

    def load_model(self):
        data = np.load('cnn_weights.npz')
        self.conv_filters = data['conv_filters']
        self.layer_weights = data['layer_weights']
        self.layer_biases = data['layer_biases']

        #return conv_filters, layer_weights, layer_biases

    def predict(self, image):
        _, _, _, _, probs = self.forward(image)
        return np.argmax(probs)

    def accuracy(self, images, labels):
        self.load_model()
        correct = 0
        n = images.shape[0]
        for i in range(n):
            image = images[i]
            label = labels[i]

            if self.predict(image) == label:
                correct += 1

        print(f'Accuracy: {correct/n:.4f}')

(image_train, label_train), (image_test, label_test) = tf.keras.datasets.mnist.load_data()

image_train = image_train.astype(np.float32) / 255.0
image_test = image_test.astype(np.float32) / 255.0
image_size = image_train[0].shape[0]

num_filters = 8
filter_size = 5

output_height = image_size - filter_size + 1
output_width = image_size - filter_size + 1

flat_size = num_filters * ((image_size - filter_size + 1) ** 2)

learning_rate = 0.01

model = NeuralNetwork()

if __name__ == '__main__':
    start_time = time.time()
    #model.train(image_train[:10000], label_train[:10000], 3)
    model.accuracy(image_test[0], label_test[0])

    end_time = time.time()
    print(f'Runtime: {end_time - start_time :.4f}')

