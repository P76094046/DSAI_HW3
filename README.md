# DSAI_HW3

### Data preprocessing 
- 首先計算全部 50 個 target 資料 generation 以及 consumption 兩者的平均值和標準差。
- 訓練時，在 data loader 中每次讀入一個 target 的資料就先減掉此平均值和除以標準差，得到歸一化後的數據才會放入模型訓練。
- 測試集也需歸一化，但使用的是訓練集的平均值和標準差。


### Model: LSTM





### Bidding strategy




