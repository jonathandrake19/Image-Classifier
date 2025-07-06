import tensorflow as tf
import numpy as np

(image_train, label_train), (image_test, label_test) = tf.keras.datasets.mnist.load_data()

image_train = image_train.astype(np.float32) / 255.0
image_test = image_test.astype(np.float32) / 255.0

num_filters = 8
filter_size = 5
image_size = image_train[0].shape[0]
learning_rate = 0.01

def forward(image):
    global conv_filters, layer_weights, layer_biases
    out_height = image_size - filter_size + 1
    out_width = image_size - filter_size + 1

    conv_output = np.zeros((num_filters, out_height, out_width))

    for f in range(num_filters):
        for i in range(out_height):
            for j in range(out_width):
                image_area = image[i:i + filter_size, j:j + filter_size]

                conv_output[f, i, j] = np.sum(image_area * conv_filters[f])

    relu_output = np.maximum(conv_output, 0)

    output_flat = relu_output.flatten()

    layer_output = np.dot(layer_weights, output_flat) + layer_biases

    exp_layer_2 = np.exp(layer_output - np.max(layer_output))
    probs = exp_layer_2 / np.sum(exp_layer_2)

    return conv_output, relu_output, output_flat, layer_output, probs

def backward(image, label, conv_output, relu_output, output_flat, probs):
    global conv_filters, layer_weights, layer_biases, learning_rate, filters

    d_layer = probs.copy()
    d_layer[label] -= 1

    d_layer_weights = np.outer(d_layer, output_flat)
    d_layer_bias = d_layer

    d_flattened = np.dot(layer_weights.T, d_layer)
    d_relu_output = d_flattened.reshape(relu_output.shape)

    d_conv_output = d_relu_output * (conv_output > 0)

    d_conv_filter = np.zeros((num_filters, filter_size, filter_size))

    for f in range(num_filters):
        for i in range(conv_output.shape[0]):
            for j in range(conv_output.shape[1]):
                area = image[i:i+filter_size, j:j+filter_size]
                d_conv_filter[f] += d_conv_output[f, i, j] * area

    layer_weights -= learning_rate * d_layer_weights
    layer_biases -= learning_rate * d_layer_bias
    conv_filters -= learning_rate * d_conv_filter

    epsilon = 1e-9
    loss = -np.log(probs[label] + epsilon)

    return loss

def train(images, labels, epochs):
    conv_filters = np.random.rand(num_filters, filter_size, filter_size) * 0.1

    flat_size = num_filters * ((image_size - filter_size + 1) ** 2)
    layer_weights = np.random.rand(10, flat_size) * 0.1
    layer_biases = np.random.rand(10)

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        for i in range(images.shape[0]):
            image = images[i]
            label = labels[i]

            conv_output, relu_output, flat, layer, probs = forward(image)

            loss = backward(image, label, conv_output, relu_output, flat, probs)
            total_loss += loss

            if np.argmax(probs) == label:
                correct += 1
            
            if (i + 1) % 1000 == 0:
                print(f"Image {i+1}, Avg Loss: {total_loss/(i+1):.4f}, Accuracy: {correct/(i+1):.4f}")

        print(f"Epoch {epoch+1} completed. Avg Loss: {total_loss/len(images):.4f}, Accuracy: {correct/len(images):.4f}")

    np.savez('cnn_weights.npz', conv_filters = conv_filters, layer_weights = layer_weights, layer_biases = layer_biases)

def load_model():
    data = np.load('cnn_weights.npz')
    conv_filters = data['conv_filters']
    layer_weights = data['layer_weights']
    layer_biases = data['layer_biases']

    return conv_filters, layer_weights, layer_biases

def predict(image):
    _, _, _, _, probs = forward(image)
    return np.argmax(probs)

def accuracy(images, labels):
    correct = 0
    n = images.shape[0]
    for i in range(n):
        image = images[i]
        label = labels[i]

        if predict(image) == label:
            correct += 1

    print(f'Accuracy: {correct/n:.4f}')

#train(image_train[:10000], label_train[:10000], 3)

conv_filters, layer_weights, layer_biases = load_model()
#accuracy(image_test[:1000], label_test[:1000])

