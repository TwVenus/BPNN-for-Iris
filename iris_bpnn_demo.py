import random
import numpy as np
import pandas as pd

class Readfile(object):
    def __init__(self, data_path):
        self.file = pd.read_csv(data_path, header=None)
        self.feature_list = np.array(self.file.iloc[0:, [0, 1, 2, 3]].values)
        for i in range(0, len(self.feature_list[1])):
            self.feature_list[:, i] = (self.feature_list[:, i] - self.feature_list[:, i].mean()) / self.feature_list[:, i].std()
        self.output_list = np.array(self.file.iloc[0:, 4].values)
        self.output_list = np.where(self.output_list == 'Iris-setosa', 0, self.output_list)
        self.output_list = np.where(self.output_list == 'Iris-versicolor', 1, self.output_list)
        self.output_list = np.where(self.output_list == 'Iris-virginica', 2, self.output_list)

class Bpnn(object):
    def __init__(self, dataset, learning_rate, bias, hidden_node, output_node, correct_rate, error_value, momentum):
        self.feature_list = dataset.feature_list
        self.output_list = dataset.output_list
        self.learning_rate = learning_rate
        self.correct_rate = correct_rate
        self.error_value = error_value
        self.hidden_node = hidden_node
        self.output_node = output_node
        self.momentum = momentum
        self.bias = bias

    def train(self):
        # 初始化
        self.weight_list_h = []
        self.weight_list_o = []
        # (input_node + bias) * hidden_node , input 到 hidden 的 weight
        weight_length_h = (self.feature_list.shape[1] + 1) * self.hidden_node
        for i in range(0, weight_length_h):
            self.weight_list_h.append(round(random.uniform(0.05, -0.05), 2))

        # (hidden_node + bias) * output_node ,　hidden 到 output 的 weight
        weight_length_o = (self.hidden_node + 1) * self.output_node
        for i in range(0, weight_length_o):
            self.weight_list_o.append(round(random.uniform(0.05, -0.05), 2))
        
		self.run_count = 0
        while True:
            self.pass_count = 0
            self.mse = 0
            for dataset_num in range(0, self.feature_list.shape[0]):
                ### 前饋階段
                hidden_after_formula1 = []
                # 計算input到hidden的結果
                for i in range(0, self.hidden_node):
                    hiddens_num = 0
                    for j in range(0, len(self.feature_list[1])):
                        # i * 5(self.feature_list.shape[1] + 1) + j
                        hiddens_num += self.feature_list[dataset_num][j] * self.weight_list_h[i * (len(self.feature_list[1]) + 1) + j + 1]
                    hiddens_num += self.bias * self.weight_list_h[i * (len(self.feature_list[1]) + 1)]
                    hidden_after_formula1.append(1/(1 + np.math.exp(hiddens_num*-1)))

                output_after_formula1 = []
                output_after_formula1_t = []
                for i in range(0, self.output_node):
                    output_num = 0
                    for j in range(0, self.hidden_node):
                        # i * 4(self.hidden_node + 1) + j
                        output_num += hidden_after_formula1[j] * self.weight_list_o[i * (len(hidden_after_formula1) + 1) + j + 1]
                    output_num += self.bias * self.weight_list_o[i * (len(hidden_after_formula1) + 1)]
                    output_after_formula1.append(1/(1 + np.math.exp(output_num*-1)))
                output_after_formula1_t = np.where(np.array(output_after_formula1) > 0.5, 1, 0)

                self.error = 0
                self.error_one = 0
                for i in range(0, self.output_node):
                    if self.output_list[dataset_num] == i:
                        self.error += pow(output_after_formula1[i] - 1, 2)
                        self.error_one += pow(1 - output_after_formula1_t[i], 2)
                    else:
                        self.error += pow(output_after_formula1[i], 2)
                        self.error_one += pow(0 - output_after_formula1_t[i], 2)
                self.mse += self.error/3

                ### 倒傳遞階段
                if self.error_one == 0:
                    self.pass_count += 1
                else:
                    correction_value_o = []
                    for i in range(0, self.output_node):
                        if i == self.output_list[dataset_num]:  # 表第i個是1 , 其餘是0
                            correction_value_o.append((1 - output_after_formula1[i]) * output_after_formula1[i] * (1 - output_after_formula1[i]))
                        else:
                            correction_value_o.append((0 - output_after_formula1[i]) * output_after_formula1[i] * (1 - output_after_formula1[i]))

                    correction_value_h = []
                    for i in range(0, self.hidden_node):
                        sum = 0
                        for j in range(0, self.output_node):
                            sum += self.weight_list_o[j * (len(hidden_after_formula1) + 1) + i + 1] * correction_value_o[j]
                        correction_value_h.append(hidden_after_formula1[i] * (1 - hidden_after_formula1[i])*sum)

                    # 調整input 到 hidden 的新weight
                    for i in range(0, self.hidden_node):
                        for j in range(0, len(self.feature_list[1])):
                            deltaW = correction_value_h[i] * self.feature_list[dataset_num][j]
                            self.weight_list_h[i * (len(self.feature_list[1]) + 1) + j + 1] += self.learning_rate * deltaW + self.momentum * deltaW
                        deltaW = correction_value_h[i] * self.bias
                        self.weight_list_h[i * (len(self.feature_list[1]) + 1)] += self.learning_rate * deltaW + self.momentum * deltaW

                    # 調整hidden 到 output 的新weight
                    for i in range(0, self.output_node):
                        for j in range(0, self.hidden_node):
                            deltaW = correction_value_o[i] * hidden_after_formula1[j]
                            self.weight_list_o[i * (len(hidden_after_formula1) + 1) + j + 1] += self.learning_rate * deltaW + self.momentum * deltaW
                        deltaW = correction_value_o[i] * self.bias
                        self.weight_list_o[i * (len(hidden_after_formula1) + 1)] += self.learning_rate * deltaW + self.momentum * deltaW

            self.run_count += 1
            self.print(self.pass_count/self.feature_list.shape[0], self.mse/self.feature_list.shape[0])

            ### 收斂條件
            if self.pass_count >= int(self.correct_rate*self.feature_list.shape[0]):
                self.print(self.pass_count / self.feature_list.shape[0], self.mse / self.feature_list.shape[0])
                break

    def print(self, correct_rate, mse):
        print("count :", self.run_count, " correct_rate : ", correct_rate, " mse : ", mse)


if __name__ == "__main__":
    dataset = Readfile("iris.txt")
    # learning_rate = 學習速率, bias = 偏差值, hidden_node = 隱藏層node個數, output_node = 輸出層個數, correct_rate = 終止條件, error_value = 誤差數
    bpnn = Bpnn(dataset, learning_rate=0.05, bias=-1, hidden_node=3, output_node=3, correct_rate=0.98, error_value=0.001, momentum=0.01)
    bpnn.train()