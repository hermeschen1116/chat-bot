# 專題
中央大學專題

### HOW-TO
`run_EP.ipynb` : 測試用筆記本，可略過
`fine_tune.py` : 訓練用
`inference.py` : 和未訓練模型比較 f1-score 和 acc

---
### 結果 5 epoches

```
1h 45m 57s

Fine-tuned:
F1-score: 0.7955796269870956
Accuracy: 0.8073643410852713

Original:
F1-score: 0.06338404956666807
Accuracy: 0.10904392764857881
```

### debug

#### sent = self._sock.send(data) BrokenPipeError: [Errno 32] Broken pipe
不小心 wandb.init 兩次 笑死