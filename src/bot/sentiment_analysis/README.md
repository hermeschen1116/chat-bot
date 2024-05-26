# 專題
中央大學專題

### HOW-TO
- `run_SA.ipynb` : 測試用筆記本，可略過
- `fine_tune.py` : 訓練用
- `inference.py` : 和未訓練模型比較 f1-score 和 acc (on test dataset)
- `sweep.py` : 利用 wandb sweep agent 來找出最佳 hyperparameters

---
### 結果

Time spent: `55m 2s`
```
Fine-tuned: (full dataset 10 epoches)
              precision    recall  f1-score   support

     neutral     0.8958    0.9089    0.9023      6321
       anger     0.2642    0.1186    0.1637       118
     disgust     0.2105    0.0851    0.1212        47
        fear     0.1111    0.0588    0.0769        17
   happiness     0.5731    0.5770    0.5751      1019
     sadness     0.3158    0.1765    0.2264       102
    surprise     0.3497    0.4914    0.4086       116

    accuracy                         0.8304      7740
   macro avg     0.3886    0.3452    0.3535      7740
weighted avg     0.8220    0.8304    0.8251      7740
```
```
Fine-tuned-half: (half neutral dataset 5 epoches)
              precision    recall  f1-score   support

     neutral     0.8967    0.9073    0.9019      6321
       anger     0.2963    0.1356    0.1860       118
     disgust     0.0833    0.0213    0.0339        47
        fear     0.1250    0.0588    0.0800        17
   happiness     0.5598    0.5829    0.5712      1019
     sadness     0.3469    0.1667    0.2252       102
    surprise     0.3375    0.4655    0.3913       116

    accuracy                         0.8292      7740
   macro avg     0.3779    0.3340    0.3414      7740
weighted avg     0.8209    0.8292    0.8238      7740
```
```
Original:
              precision    recall  f1-score   support

     neutral     0.8784    0.8526    0.8653      6321
       anger     0.2124    0.3475    0.2637       118
     disgust     0.1494    0.2766    0.1940        47
        fear     0.0621    0.5294    0.1111        17
   happiness     0.6045    0.3690    0.4583      1019
     sadness     0.1349    0.3824    0.1995       102
    surprise     0.2082    0.4828    0.2909       116

    accuracy                         0.7652      7740
   macro avg     0.3214    0.4629    0.3404      7740
weighted avg     0.8061    0.7652    0.7794      7740
```
```
true: [0, 6, 0, 0, 0, 0, 0, 0, 0] 
fine-tuned: [0, 0, 0, 0, 0, 0, 1, 6, 0] 
fine-tuned-half: [0, 0, 0, 0, 0, 0, 1, 6, 0] 
original: [0, 0, 0, 0, 0, 0, 1, 6, 1]
```

### ISSUES
1. **The model did not return a loss from the inputs, only the following keys: logits. For reference, the inputs it received are input_ids,attention_mask.**


    如果沒有 `df.rename({'utterance': 'text', 'emotions': 'label'}, axis=1)` [好像會讓他讀不到，一定要是這個形式。](https://discuss.huggingface.co/t/the-model-did-not-return-a-loss-from-the-inputs-only-the-following-keys-logits-for-reference-the-inputs-it-received-are-input-values/25420/13)

2. **Unable to create tensor, you should probably activate truncation and/or padding with 'padding=True' 'truncation=True' to have batched tensors with the same length. Perhaps your features (``label`` in this case) have excessive nesting (inputs type ``list`` where type ``int`` is expected).**
   
    這就忘了我的 label 是英文(anger, disgust...)

### CONTRIBUTION ?
#### TL;DR
在 [DailyDialog](https://huggingface.co/datasets/benjaminbeilharz/better_daily_dialog) 上微調 [emotion_text_classifier](https://huggingface.co/michellejieli/emotion_text_classifier)。
> ACC 從 ~0.66 提升至 ~81

#### 整體流程
* 利用 huggingface 的 load_dataset 來直接存取資料集，並且整理成可用的形式。例如更改 feature 名稱來對應需求。
* 使用 AutoTokenizer 以及 AutoModelForSequenceClassification 來產生需要的組件，並且使用 trainer.train() 進行訓練。