import os
import torch
import wandb
from datasets import load_dataset, Dataset
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()
login(token=os.environ.get("HF_TOKEN", ""), add_to_git_credential=True)
wandb.login(key=os.environ.get("WANDB_API_KEY", ""))

wandb_config = {
    "base_model": "michellejieli/emotion_text_classifier",
}
wandb.init(
    job_type="fine-tuning",
    config=wandb_config,
    project="emotion-chat-bot-ncu",
    group="emotion_predictor_ex1",
    mode="online",
    # resume="auto"
)

base_model = "michellejieli/emotion_text_classifier"
new_model = "etc_on_dd"

def preprocessing(data):
    data = data.rename_column("utterance", "text")
    data = data.rename_column("emotion", "label") 
    data = data.remove_columns("turn_type")
    return data

def shifting_train(data):
    df = data["train"].to_pandas()
    df["label"] = df.groupby('dialog_id')["label"].shift(-1)
    df.dropna(inplace = True)
    df["label"]  = df["label"].astype(int)
    modified_dataset = Dataset.from_pandas(df)
    data["train"] = modified_dataset
    return data

data_name = "benjaminbeilharz/better_daily_dialog"
data_raw = load_dataset(data_name, num_proc=16)
data_raw = preprocessing(data_raw)
data_raw = shifting_train(data_raw)
data = data_raw

tokenizer = AutoTokenizer.from_pretrained(base_model)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True)

emotions = data

tokens2ids = list(zip(tokenizer.all_special_tokens, tokenizer.all_special_ids))
data = sorted(tokens2ids, key=lambda x: x[-1])

emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)
print(emotions_encoded["train"].column_names)

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

label2id = {
    "neutral": 0,
    "anger": 1,
    "disgust": 2,
    "fear": 3,
    "happiness": 4,
    "sadness": 5,
    "surprise": 6
}

model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=num_labels, id2label=id2label, label2id=label2id)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

batch_size = 64
logging_steps = len(emotions_encoded["train"]) // batch_size

training_args = TrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs=5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=500,
    logging_steps=logging_steps,
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to=["wandb"],
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": True},
    evaluation_strategy="epoch",
    log_level="error",
    overwrite_output_dir=True
)
wandb.config["trainer_arguments"] = training_args.to_dict()

trainer = Trainer(model=model, args=training_args,
                  compute_metrics=compute_metrics,
                  train_dataset=emotions_encoded["train"],
                  eval_dataset=emotions_encoded["validation"],
                  tokenizer=tokenizer)
trainer.train();
wandb.finish()

trainer.model.save_pretrained(new_model)