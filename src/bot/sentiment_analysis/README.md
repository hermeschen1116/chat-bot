# 專題
中央大學專題

### HOW-TO
`run_SA.ipynb` : 測試用筆記本，可略過
`fine_tune.py` : 訓練用
`inference.py` : 和未訓練模型比較 f1-score 和 acc (on test dataset)

---
### 結果 10 epoches

```
1h 15m 2s

Fine-tuned:
F1-score: 0.8237113168147658
Accuracy: 0.8468992248062015

Original:
F1-score: 0.7794121109785516
Accuracy: 0.7652454780361757
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