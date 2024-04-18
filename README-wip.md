# 專題
中央大學專題

# Usage

訓練執行 `run_generate_response_finetuning.ipynb`

拉資料執行 `test_on_data.ipynb`

# Progress

* wandb is working.
* the whole thing seemed to be working. lol
* base model changed to "meta-llama/Llama-2-7b-chat-hf"
* flash_attention_2
* takes roughly 11G vram
* use openAI API to generate dataset and trained on it
* downgraded to transformers v4.38.2
  
# To-Do

* apply_chat_template
* more data (only one emotion now)

# ISSUES

1. **OOM on 3090**
    
    add [gradient_checkpointing](https://huggingface.co/docs/transformers/v4.20.1/en/perf_train_gpu_one#using-accelerate)

    從`run_generate_response(archived)`重新來過，不知道怎麼搞的可以成功跑起來。

2. **資料集建立**
   
    利用 chatGPT 來建立資料及。回應候選以 text generation 的模式生成，一次一個回應。

3. **top_k_top_p_filtering import error**

    把 transformers 降低版本到 4.38.2 暫時解決
    [ref](https://github.com/huggingface/trl/issues/1409#issuecomment-1986880442)

    不過用舊版的疑似會對新套件有支援性問題，目前沒碰上

4. **Need a faster way to generate training data**

    現在用 chatGPT 4.0 跑，目前只跑其中一種情緒的一百筆，還有七種要跑。

# To-Do

- [ ] changing datasets
- [ ] test performance
- [ ] ...

