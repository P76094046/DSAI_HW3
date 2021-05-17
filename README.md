# DSAI_HW3
### 說明
需下載 model.hdf5 和 model2.hdf5，執行 python main.py --generation generation.csv --consumption consumption.csv --output output.csv
(training.py 為訓練模型的檔案)

### Data preprocessing 
- 首先計算全部 50 個 target 資料 generation 以及 consumption 兩者的平均值和標準差。
- 訓練時，在 data loader 中每次讀入一個 target 的資料就先減掉此平均值和除以標準差，得到歸一化後的數據才會放入模型訓練。
- 測試集也需歸一化，但使用的是訓練集的平均值和標準差。


### Model: LSTM
- 用兩個模型分別訓練 generation 和 consumption。
- 使用 168 筆資料預測 1 筆
- 參數：hidden_dim = 16, num_layers = 2, batch_size = 16, num_epochs = 50



### Bidding strategy
- 如果預測出來 generation >= consumption，那就用3元賣電。
- 如果預測出來 generation < consumption，那就用2.5元買電。



