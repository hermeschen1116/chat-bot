#Sweep to find best parameters via wandb#
import os
import random
from dotenv import load_dotenv

import torch
from datasets import load_dataset
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
    )
from huggingface_hub import login
from peft import LoraConfig, get_peft_model
import wandb

load_dotenv()
login(token=os.environ.get("HF_TOKEN", ""), add_to_git_credential=True)
wandb.login(key=os.environ.get("WANDB_API_KEY", ""))

base_model = "michellejieli/emotion_text_classifier"
# new_model = "./etc_on_dd"

def preprocessing(data):
    data = data.rename_column("utterance", "text")
    data = data.rename_column("emotion", "label")
    data = data.remove_columns(["dialog_id", "turn_type"])
    return data

def remove_half_train(data):
    data_set = data["train"]
    label_0_indices = [i for i, row in enumerate(data_set) if row['label'] == 0]
    num_to_remove = len(label_0_indices) // 2
    indices_to_remove = random.sample(label_0_indices, num_to_remove)
    filtered_data = data_set.filter(lambda x, i: i not in indices_to_remove, with_indices=True)
    data["train"] = filtered_data
    return data

data_name = "benjaminbeilharz/better_daily_dialog"
data_raw = load_dataset(data_name, num_proc=16)
data_raw = preprocessing(data_raw)
data_raw = remove_half_train(data_raw)
data = data_raw

tokenizer = AutoTokenizer.from_pretrained(base_model)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True)

emotions = data

tokens2ids = list(zip(tokenizer.all_special_tokens, tokenizer.all_special_ids))
data = sorted(tokens2ids, key=lambda x: x[-1])

emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_labels = 7
id2label = {
    0: "neutral",
    1: "anger",
    2: "disgust",
    3: "fear",
    4: "happiness",
    5: "sadness",
    6: "surprise"
}
label2id = {v: k for k, v in id2label.items()}

model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=num_labels, id2label=id2label, label2id=label2id)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    wandb.log({"eval_f1": f1})
    return {"accuracy": acc, "f1": f1}

sweep_config = {
    "method": "random",
    "name": "random-sweep",
    "metric": {
        "goal": "maximize",
        "name": "eval_f1"
    },
    "parameters": {
        "epochs": {
            "value": 5
        },
        "batch_size": {
            "values": [8, 16, 32]
        },
        "learning_rate": {
            "values": [0.0005, 0.0001, 0.00005]
        },
        "weight_decay": {
            "values": [0.0001, 0.1]
        },
        "lora_r": {
            "values": [64, 128, 256]
        },
        "lora_alpha": {
            "values": [16, 32, 64]
        },
        "lora_dropout": {
            "values": [0.0, 0.2, 0.4]
        }
    }
}

def train(config=None):
    with wandb.init(config=config):
        config = wandb.config
    
    training_args = TrainingArguments(
        output_dir="./sweeps-SA",
	    report_to='wandb',
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        save_strategy='epoch',
        evaluation_strategy='epoch',
        logging_strategy='epoch',
        load_best_model_at_end=True,
        remove_unused_columns=False,
        save_total_limit=5,
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True}
    )

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="SEQ_CLS",
        use_rslora = True
    )
    
    peft_model = get_peft_model(model, lora_config)

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=emotions_encoded["train"],
        eval_dataset=emotions_encoded["validation"],
        compute_metrics=compute_metrics,
    )
    
    trainer.train()

sweep_id = wandb.sweep(sweep_config, project='sentiment-analysis-sweeps')
wandb.agent(sweep_id, train, count=100)
wandb.finish()