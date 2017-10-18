# business_intelligence_BPNN
## 網路架構 - input, output, hidden
![](https://i.imgur.com/lbURdBs.png)

## 使用方法
### 輸入下列語法執行
```` iris_bpnn_demo.py````

## 參數設定
#### 1. learning_rate = 學習速率 
#### 2. error_value = 誤差值 
#### 3. hidden_node = 隱藏層node個數 
#### 4. output_node = 輸出層個數 
#### 5. correct_rate = 終止條件 
#### 6. bias = 偏差值 

## 收斂條件
```` 將 output 做 one_hot_encode 之後，與期望輸出做差方和並平均，如果小於 error_value 表示該筆資料分對，值到150筆資料完全分對就停止 ````

## 執行結果 
#### 正確率：100% ####
