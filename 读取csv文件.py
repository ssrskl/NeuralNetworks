import numpy
import matplotlib.pyplot as plt

data_file = open("data/mnist_test_10.csv", 'r')
data_list = data_file.readlines()
for i in range(5):
    all_values = data_list[i].split(',')
    image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
    print(image_array.shape)
    plt.imshow(image_array, cmap='Greys', interpolation='None')
    plt.show()
