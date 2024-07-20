# This is a mixed demo from https://huggingface.co/blog/zh/putting_rl_back_in_rlhf_with_rloo and trl example

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from datasets import load_dataset
from trl.trainer.rloo_trainer import RLOOConfig, RLOOTrainer
from trl.trainer.utils import SIMPLE_QUERY_CHAT_TEMPLATE

base_model_name = "EleutherAI/pythia-1b-deduped"
tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_side="left")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
if tokenizer.chat_template is None:
    tokenizer.chat_template = SIMPLE_QUERY_CHAT_TEMPLATE
reward_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=1)
ref_policy = AutoModelForCausalLM.from_pretrained(base_model_name)
policy = AutoModelForCausalLM.from_pretrained(base_model_name)

raw_datasets = load_dataset("trl-internal-testing/tldr-preference-sft-trl-style")
for key in raw_datasets:
    raw_datasets[key] = raw_datasets[key].select(range(1000))
train_dataset = raw_datasets["train"]
eval_dataset = raw_datasets["validation"]

def prepare_dataset(dataset, tokenizer):
    """pre-tokenize the dataset before training; only collate during training"""

    def tokenize(element):
        input_ids = tokenizer.apply_chat_template(
            element["messages"][:1],
            padding=False,
            add_generation_prompt=True,
        )
        return {"input_ids": input_ids, "lengths": len(input_ids)}

    return dataset.map(
        tokenize,
        remove_columns=dataset.column_names,
        num_proc=1 if True else multiprocessing.cpu_count(),
        load_from_cache_file=not True,
    )

train_dataset = prepare_dataset(train_dataset, tokenizer)
eval_dataset = prepare_dataset(eval_dataset, tokenizer)
# filtering
train_dataset = train_dataset.filter(lambda x: x["lengths"] <= 512)

trainer = RLOOTrainer(
    config=RLOOConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=64,
        total_episodes=30000,
        output_dir = "./test"
    ),
    tokenizer=tokenizer,
    policy=policy,
    ref_policy=ref_policy,
    reward_model=reward_model,
    train_dataset=train_dataset,
)
trainer.train()
