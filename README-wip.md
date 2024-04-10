# 專題
中央大學專題

### HOW-TO
執行 `run_generate_response.ipynb`

### ISSUES
1. **OOM on 3090**
    
    add [gradient_checkpointing](https://huggingface.co/docs/transformers/v4.20.1/en/perf_train_gpu_one#using-accelerate)

2. **資料集建立**
   
    利用 chatGPT 來建立。回應以 text generation 的模式生成，一次一個回應情緒。

### CONTRIBUTION ?


#### 整體流程



### To-Do
- [ ] changing datasets
- [ ] test performance
- [ ] ...
---
### 專題作業時程
#### Sentiment Analysis （3月初）
* Test prompt and optimize 
#### Response Generator （3月底）
* Test prompt and optimize
#### Candidate Generator （4月中）
* Test prompt
* Test different length of history
* Optimize 
* (option) use Trl to strengthen divergence 
#### Similarity Analysis （3月底）
* Find right math formula
* Experiment 
#### Emotion Model （4月底）
* Understand different types of attention mechanism 
* Build model with attention mechanism 
* Optimize
#### Full Model（5月中）
* Optimize and Improve 
* Application

