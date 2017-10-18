import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Readfile(object):
    def __init__(self, data_path):
        self.file = pd.read_csv(data_path, header=None)
        self.feature_list = np.array(self.file.iloc[0:, [0, 1, 2, 3]].values)
        self.output_list = np.array(self.file.iloc[0:, 4].values)
        self.output_list = np.where(self.output_list == 'Iris-setosa', 0, self.output_list)
        self.output_list = np.where(self.output_list == 'Iris-versicolor', 1, self.output_list)
        self.output_list = np.where(self.output_list == 'Iris-virginica', 2, self.output_list)

class Bpnn(object):
    def __init__(self, dataset, learning_rate=0.01, bias=-1, hidden_node=3, output_node=3):
        self.feature_list = dataset.feature_list
        self.output_list = dataset.output_list
        self.learning_rate = learning_rate
        self.hidden_node = hidden_node
        self.output_node = output_node
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

        # while True:
        ### 前饋階段
        for dataset_num in range(0, self.feature_list.shape[0]):
            pass_count = 0
            hidden_after_formula1 = []
            # 計算input到hidden的結果
            for i in range(0, self.hidden_node):
                hiddens_num = 0
                for j in range(0, len(self.feature_list[1])):
                    # i * 5(self.feature_list.shape[1] + 1) + j
                    hiddens_num += self.feature_list[dataset_num][j] * self.weight_list_h[i * (len(self.feature_list[1]) + 1) + j + 1]
                hiddens_num += -1 * self.weight_list_h[i*(len(self.feature_list[1]) + 1)]
                hidden_after_formula1.append(hiddens_num)

            output_after_formula1 = []
            for i in range(0, self.output_node):
                output_num = 0
                for j in range(0, len(hidden_after_formula1)):
                    # i * 4(self.hidden_node + 1) + j
                    output_num += hidden_after_formula1[j] * self.weight_list_o[i * (len(hidden_after_formula1) + 1) + j + 1]
                output_num += -1 * self.weight_list_o[i * (len(hidden_after_formula1) + 1)]
                output_after_formula1.append(output_num)

            error = 0
            for i in range(0, self.output_node):
                if self.output_list[dataset_num] == i:
                    for j in range(0, len(output_after_formula1)):
                        if j == i:  # 表第i個是1 , 其餘是0
                            error += pow(output_after_formula1[j] - 1, 2)
                        else:
                            error += pow(output_after_formula1[j], 2)

            ### 倒傳遞階段
            if error < 0.001:
                pass_count += 1
            else:
                correction_value_o = []
                for i in range(0, self.output_node):
                    if i == self.output_list[dataset_num]:  # 表第i個是1 , 其餘是0
                        correction_value_o.append((1 - output_after_formula1[i]) * output_after_formula1[i] * (1 - output_after_formula1[i]))
                    else:
                        correction_value_o.append((0 - output_after_formula1[i]) * output_after_formula1[i] * (1 - output_after_formula1[i]))

                correction_value_h = []
                for i in range(0, self.hidden_node):
                    corrections = 0
                    for j in range(0, self.output_node):
                        corrections += self.weight_list_o[j * (len(hidden_after_formula1) + 1) + i + 1] * correction_value_o[j]
                    correction_value_h.append(hidden_after_formula1[i] * (1 - hidden_after_formula1[i])*corrections)



if __name__ == "__main__":
    dataset = Readfile("iris.txt")
    bpnn = Bpnn(dataset, learning_rate=0.01, bias=-1, hidden_node=3, output_node=3)
    bpnn.train()