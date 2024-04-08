import argparse
import os

import huggingface_hub
import torch
from accelerate import DataLoaderConfiguration

import wandb
from datasets import load_dataset
from dotenv import load_dotenv
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# commandline inputs
parser = argparse.ArgumentParser(prog="Fine-Tuning", description="Fine-Tuning Script For Response Generator")
parser.add_argument("--base_model", required=True, type=str)
parser.add_argument("--tokenizer", required=False, type=str, default=None)
parser.add_argument("--name_or_path_for_fine_tuned_model", "-n", required=False, type=str, default="")
parser.add_argument("--experiment_detail", required=True, type=str)
parser.add_argument(
    "--wandb_mode",
    required=False,
    type=str,
    choices=["online", "offline", "disabled"],
    default="online"
)
parser.add_argument("--num_epochs", required=False, type=int, default=1)
parser.add_argument("--enable_flash_attention_2", required=False, type=bool, default=True)
parser.add_argument("--system_prompt", required=False, type=str, default="")
parser.add_argument("--chat_template_file", required=True, type=str, default="")

arguments = parser.parse_args()
arguments.tokenizer = arguments.base_model if arguments.tokenizer is None else arguments.tokenizer
chat_template: dict = eval(open(arguments.chat_template_file, 'r', encoding='utf-8', closefd=True).read())

load_dotenv(encoding="utf-8")
huggingface_hub.login(token=os.environ.get("HF_TOKEN", ""), add_to_git_credential=True)
wandb.login(key=os.environ.get("WANDB_API_KEY", ""), relogin=True)

# Initialize Wandb
wandb_config = {
    "base_model": arguments.base_model,
    "tokenizer": arguments.tokenizer,
    "name_or_path_for_fine_tuned_model": arguments.name_or_path_for_fine_tuned_model,
    "system_prompt": arguments.system_prompt,
    "chat_template": chat_template["chat"],
    "instruction_template": chat_template["instruction"],
    "response_template": chat_template["response"]
}
wandb.init(
    job_type="fine-tuning",
    config=wandb_config,
    project="emotion-chat-bot-ncu",
    group="Response Generator",
    notes=arguments.experiment_detail,
    mode=arguments.wandb_mode,
    resume="auto"
)

# Load Dataset
dataset = load_dataset("daily_dialog",
                       split="train[:1]+validation[:1]",
                       num_proc=16,
                       trust_remote_code=True).remove_columns("act")
dataset = dataset.rename_column("emotion", "emotion_id")
emotion_labels: list = dataset.features["emotion_id"].feature.names
emotion_labels[0] = "neutral"
dataset = dataset.map(lambda samples: {
    "emotion": [[emotion_labels[emotion_id] for emotion_id in sample] for sample in samples]
}, input_columns="emotion_id", remove_columns="emotion_id", batched=True, num_proc=16)
dataset = dataset.map(lambda samples: {
    "emotion": [sample[:-1] if len(sample) % 2 == 1 else sample for sample in samples]
}, input_columns="emotion", batched=True, num_proc=16)
dataset = dataset.map(lambda samples: {
    "dialog": [[dialog.strip() for dialog in sample] for sample in samples]
}, input_columns="dialog", batched=True, num_proc=16)
dataset = dataset.map(lambda samples: {
    "dialog": [sample[:-1] if len(sample) % 2 == 1 else sample for sample in samples]
}, input_columns="dialog", batched=True, num_proc=16)
dataset = dataset.map(lambda samples: {
    "prompt": [[{
                    "role": "user" if i % 2 == 0 else "assistant",
                    # "content": {"emotion": emotion, "dialog": dialog}
                    "content": dialog
                }
                for i, (emotion, dialog) in enumerate(zip(sample[0], sample[1]))]
               for sample in zip(samples["emotion"], samples["dialog"])]
}, remove_columns=["emotion", "dialog"], batched=True, num_proc=16)
# dataset = dataset.map(lambda samples: {
#     "prompt": [[{
#                     "role": "system",
#                     "content": arguments.system_prompt
#                 }] + sample
#                for sample in samples]
# }, input_columns="prompt", batched=True, num_proc=16)
# dataset = dataset.train_test_split(test_size=0.2)

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(arguments.tokenizer)
tokenizer.eos_token = "<eos>" if tokenizer.eos_token is None else tokenizer.eos_token
tokenizer.pad_token = "<pad>" if tokenizer.pad_token is None else tokenizer.pad_token
tokenizer.padding_side = "right"
tokenizer.clean_up_tokenization_spaces = True


def prompt_compose(sample: str) -> str:
    return tokenizer.apply_chat_template(sample,
                                         # chat_template=chat_template["chat"],
                                         add_generation_prompt=True,
                                         tokenize=False,
                                         padding=True,
                                         max_length=4096,
                                         return_tensors="pt"
                                         )


dataset = dataset.map(lambda sample: {
    "prompt": prompt_compose(sample)
}, input_columns="prompt", num_proc=16)
wandb.config["example_prompt"] = dataset[0]["prompt"]
dataset = dataset.train_test_split(test_size=0.2)

# Configurations
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)
quantization_config = quantization_config if torch.cuda.is_available() else None
wandb.config["quantization_configuration"] = quantization_config.to_dict() if quantization_config is not None else {}

lora_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=8,
    bias="none",
    task_type="CAUSAL_LM"
)
wandb.config["lora_configuration"] = lora_config.to_dict()

trainer_arguments = TrainingArguments(
    output_dir="./checkpoints",
    overwrite_output_dir=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,
    eval_accumulation_steps=1,
    learning_rate=2e-4,
    weight_decay=0.001,
    max_grad_norm=0.3,
    num_train_epochs=arguments.num_epochs,
    lr_scheduler_type="constant",
    warmup_ratio=0.03,
    max_steps=-1,
    logging_steps=25,
    save_steps=25,
    save_total_limit=5,
    jit_mode_eval=True,
    bf16=False,
    fp16=False,
    dataloader_num_workers=16,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    optim="paged_adamw_32bit",
    group_by_length=True,
    report_to=["wandb"],
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": True},
    auto_find_batch_size=True,
    torch_compile=True
)
wandb.config["trainer_arguments"] = trainer_arguments.to_dict()

device_map: str = "auto" if torch.cuda.is_available() else "cpu"
flash_attention: str = "flash_attention_2" if arguments.enable_flash_attention_2 else None
# Load Model
base_model = AutoModelForCausalLM.from_pretrained(
    arguments.base_model,
    quantization_config=quantization_config,
    attn_implementation=flash_attention,
    pretraining_tp=1,
    use_cache=False,
    device_map=device_map,
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

data_collator = DataCollatorForCompletionOnlyLM(
    chat_template["response"],
    instruction_template=chat_template["instruction"],
    tokenizer=tokenizer
)

# Setup Tuner
tuner = SFTTrainer(
    model=base_model,
    args=trainer_arguments,
    # data_collator=data_collator,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=lora_config,
    dataset_text_field="prompt",
    tokenizer=tokenizer,
    max_seq_length=4096,
    dataset_num_proc=16
)

tuner.train()

wandb.finish()
