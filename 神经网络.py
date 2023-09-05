import numpy
import scipy.special


class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        # 创建第一层的链接权重矩阵
        self.wih = (numpy.random.rand(self.hnodes, self.inodes) - 0.5)
        # 创建第二层的链接权重矩阵
        self.who = (numpy.random.rand(self.onodes, self.hnodes) - 0.5)
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    # 训练阶段只有两个任务，第一个是query也就是查询阶段，即正常的输出阶段，第二个反向传播阶段，即调整链接权重的阶段。
    def train(self, input_list, targets_list):
        # 将输入的一维数组转换为2维矩阵，并转置
        inputs = numpy.array(input_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # 下面的一段代码是正向传播的过程
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 对第一层的输出矩阵进行激活函数处理
        hidden_outputs = self.activation_function(hidden_inputs)
        # 将第二层的权重矩阵与第二层的输入矩阵相乘得到第二层的输出矩阵
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 对第二层的输出矩阵进行激活函数处理
        final_outputs = self.activation_function(final_inputs)

        # 通过计算输出层的误差来计算隐藏层的误差
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)
        # 调整权重
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))

        pass

    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        # 将第一层的链接权重矩阵与输入矩阵相乘得到第一层的输出矩阵
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 对第一层的输出矩阵进行激活函数处理
        hidden_outputs = self.activation_function(hidden_inputs)
        # 将第二层的权重矩阵与第二层的输入矩阵相乘得到第二层的输出矩阵
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 对第二层的输出矩阵进行激活函数处理
        final_outputs = self.activation_function(final_inputs)
        return final_outputs


input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.1
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
# 导入训练数据
# trainimg_data_file = open("data/mnist_train_100.csv", 'r')
trainimg_data_file = open("data/mnist_train.csv", 'r')
trainimg_data_list = trainimg_data_file.readlines()
trainimg_data_file.close()
# 对权重进行训练
for record in trainimg_data_list:
    all_values = record.split(',')
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    targets = numpy.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)
    pass
# 数据测试
test_data_file = open("data/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()
scorecard = []
for record in test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    print("正确值:" + str(correct_label))
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    outputs = n.query(inputs)
    label = numpy.argmax(outputs)
    print("神经网络结果:" + str(label))
    # 检查正确率
    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)

print("正确率:" + str(sum(scorecard) / len(scorecard)))
