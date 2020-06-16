import numpy as np


class PerceptronMultiClass():


    def __init__(self, no_of_inputs, no_of_outputs, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.no_of_inputs = no_of_inputs + 1 
        self.no_of_outputs = no_of_outputs
        self.weights = np.random.rand(self.no_of_inputs, self.no_of_outputs)

    # 일단은 한개씩 입력을 처리하고 있다.
    # 향후에는, 이것을 매트릭스의 곱으로 변경해야한다.
    def predict(self, input_data):
        def activation(x):
            #return 1.0 if x > 0.0 else -1.0
            return (1 / (1 + np.exp(-x)))

        net = np.array([activation(x) for x in np.dot(input_data, self.weights)])
        return np.argmax(net), net

    def test(self, input_datas, targets):
        correct = 0
        total = 0
        for input_data, target in zip(input_datas, targets):
            class_id, prediction = self.predict(input_data)
            total += 1
            if class_id == np.argmax(target):
                correct += 1
        print("Correct {} among {}".format(correct, total))
    
    def train(self, input_datas, targets, n_epoch):
        for epoch in range(n_epoch):
            sum_error = np.zeros(self.no_of_outputs)
            for input_data, target in zip(input_datas, targets):
                class_id, prediction = self.predict(input_data)
                if class_id != np.argmax(target):
                    error = target - prediction
                    sum_error += error
                    self.weights += (self.learning_rate * error.reshape(self.no_of_outputs,1) 
                                                        * input_data.reshape(1, self.no_of_inputs)).T

            if epoch % 1000 == 0:
                print('>', epoch, np.sum(np.abs(sum_error)))
    
    # saving  weight file
    def write_weights(self, weights_file):
        f = open(weights_file, 'w')
        f.write(str(self.weights.tolist()))
        f.close()

    def read_weights(self, weights_file):
        f = open(weights_file, 'r')
        weight_str = f.read()
        #print(weight_str)
        self.weights = eval(weight_str)
        f.close()