# image processing

##### 一、概述
```
本程式為使用python語法執行『倒傳遞演算法』來分類IRIS資料集。
資料集：https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
```

##### 二、網路架構 ( hidden 層數：1 ， hidden node 個數：3)
![](https://i.imgur.com/LbaQaRp.png)

##### 三、程式概述 
###### 1. 初始化階段
決定網路架構，初始化weight，weight用random取(-0.05 ~ 0.05)之間，將input到hidden與hidden到input的weight分開存進各別的一維陣列中。
```
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
```

###### 2. 前饋階段
在多層感知機裡，活化函數最常使用的是對數型式的「sigmoidal 函數」。
```
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
```

###### 3. 倒傳遞階段
每筆資料是否分類正確標準為：計算輸出如大於0.5當作1，反之為0，如與期望輸出相差平方加總後等於0，表分類正確，即可做下一筆，反之進入倒傳遞階段，該階段是由「delta 法則」來定義。
```
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
```

##### 4. 終止條件
當迭代次數大於10000次或者是準確率到達0.98就終止程式。
```
    if self.pass_count >= int(self.correct_rate*self.feature_list.shape[0]):
        self.print(self.pass_count / self.feature_list.shape[0], self.mse / self.feature_list.shape[0])
        break
```

##### 四、hyper parameter 調整比照表
　
![](https://i.imgur.com/ywRt5UX.jpg)

##### 五、結果
**訓練數量：150筆**　**準確率：約0.98**　**迭代次數：3148**　**MSE：約0.09486**
　

![](https://i.imgur.com/IX6Gxhx.png)

 

