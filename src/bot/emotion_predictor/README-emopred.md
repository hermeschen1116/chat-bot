# 專題
中央大學專題

### HOW-TO
`run_EP.ipynb` : 測試用筆記本，可略過
`fine_tune.py` : 訓練用
`inference.py` : 和未訓練模型比較 f1-score 和 acc (on test dataset)
`sweep.py` : 利用 `wandb sweep agent` 來找出最佳 hyperparameters

---
### 結果
```
Fine-tuned: (10 epoches)
              precision    recall  f1-score   support

     neutral     0.8159    0.9903    0.8946      5454
       anger     0.0000    0.0000    0.0000       102
     disgust     0.0000    0.0000    0.0000        41
        fear     0.0000    0.0000    0.0000        14
   happiness     0.5299    0.0669    0.1188       927
     sadness     0.0000    0.0000    0.0000        94
    surprise     0.0000    0.0000    0.0000       108

    accuracy                         0.8105      6740
   macro avg     0.1923    0.1510    0.1448      6740
weighted avg     0.7331    0.8105    0.7403      6740
```
```
Original:
              precision    recall  f1-score   support

     neutral     0.8290    0.8188    0.8239      5454
       anger     0.0970    0.1569    0.1199       102
     disgust     0.0548    0.0976    0.0702        41
        fear     0.0000    0.0000    0.0000        14
   happiness     0.3987    0.2060    0.2717       927
     sadness     0.0354    0.0957    0.0517        94
    surprise     0.0079    0.0185    0.0111       108

    accuracy                         0.6955      6740
   macro avg     0.2033    0.1991    0.1926      6740
weighted avg     0.7281    0.6955    0.7072      6740
```
```
true: [6, 0, 0, 0, 0, 0, 0, 0, 0] 
fine-tuned: [0, 0, 0, 0, 0, 0, 0, 0, 0] 
original: [0, 0, 0, 0, 0, 0, 1, 6, 1]
```
### debug

#### sent = self._sock.send(data) BrokenPipeError: [Errno 32] Broken pipe
不小心 wandb.init 兩次 笑死