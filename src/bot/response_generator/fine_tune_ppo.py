from argparse import ArgumentParser
from dataclasses import dataclass, Field

import torch
from trl.core import LengthSampler
import wandb
from bitsandbytes.optim import PagedLion32bit
from datasets import load_dataset
from peft.peft_model import PeftModel
from tqdm.auto import tqdm
from transformers import (
	BitsAndBytesConfig,
	GenerationConfig,
	HfArgumentParser,
	pipeline, TextStreamer
)
from transformers.hf_argparser import HfArg
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from unsloth import FastLanguageModel

from libs import CommonScriptArguments, CommonWanDBArguments


@dataclass
class ScriptArguments(CommonScriptArguments):
	chat_template_file: Field[str] = HfArg(aliases="--chat-template-file", default="")


config_getter = ArgumentParser()
config_getter.add_argument("--json_file", required=True, type=str)
config = config_getter.parse_args()

parser = HfArgumentParser((ScriptArguments, CommonWanDBArguments))
args, wandb_args = parser.parse_json_file(config.json_file)

chat_template: dict = eval(open(args.chat_template_file, "r", encoding="utf-8", closefd=True).read())

# Initialize Wandb
run = wandb.init(
	job_type=wandb_args.job_type,
	config=wandb_args.config,
	project=wandb_args.project,
	group=wandb_args.group,
	notes=wandb_args.notes,
	mode=wandb_args.mode,
	resume=wandb_args.resume
)
wandb.config["chat_template"] = chat_template["template"]
wandb.config["instruction_template"] = chat_template["instruction"]
wandb.config["response_template"] = chat_template["response"]
wandb.config["special_tokens"] = chat_template["special_tokens"]

# Load Dataset
dataset = load_dataset(
	"hermeschen1116/daily_dialog_for_RG",
	split="train+validation",
	keep_in_memory=True,
	num_proc=16,
	trust_remote_code=True
)
dataset = dataset.take(1024)   # use very small dataset to debug

history_length: int = 2 * wandb.config["num_turns_history"]
dataset = dataset.filter(lambda sample: len(sample) >= (2 + history_length), input_columns="prompt", num_proc=16)
print(f"dataset size after filter: {len(dataset)}")

dataset = dataset.map(lambda sample: {
	"prompt": sample[i: i + 2 + history_length]
	for i in range(0, len(sample) - 2, 2) if (i + 2 + history_length) <= len(sample)
}, input_columns="prompt", num_proc=16)

system_prompt: list = [{"role": "system", "content": {"emotion": "", "dialog": wandb.config["system_prompt"]}}]

dataset = dataset.map(lambda samples: {
	"prompt": [system_prompt + sample for sample in samples]
}, input_columns="prompt", batched=True, num_proc=16)

emotion_labels: list = ["neutral", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]

dataset = dataset.map(lambda samples: {
	"query": [
		sample[:-1] + [{"role": "assistant", "content": {"emotion": sample[-1]["content"]["emotion"], "dialog": ""}}]
		for sample in samples
	],
	"label": [emotion_labels.index(sample[-1]["content"]["emotion"]) for sample in samples]
}, input_columns="prompt", remove_columns="prompt", batched=True, num_proc=16)

# Load Tokenizer
base_model, tokenizer = FastLanguageModel.from_pretrained(
	wandb.config["base_model"],
	attn_implementation="flash_attention_2",
	pretraining_tp=1,
	load_in_4bit=True,
	use_cache=False,
	device_map="auto",
	use_gradient_checkpointing=True,
	low_cpu_mem_usage=True,
	trust_remote_code=True,
)
tokenizer.padding_side = "left"
tokenizer.clean_up_tokenization_spaces = True
tokenizer.chat_template = wandb.config["chat_template"]
tokenizer.add_special_tokens(wandb.config["special_tokens"])
base_model.resize_token_embeddings(len(tokenizer))

base_model_with_adapter = PeftModel.from_pretrained(base_model, wandb.config["adapter"])
base_model_with_adapter.print_trainable_parameters()
FastLanguageModel.for_inference(base_model_with_adapter)

ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(
	base_model_with_adapter,
	device_map="auto"
)

dataset = dataset.with_format("torch")
dataset = dataset.map(lambda sample: {
	"input_ids": tokenizer.apply_chat_template(
		sample,
		tokenize=True,
		padding="max_length",
		max_length=wandb.config["max_input_tokens"],
		add_generation_prompt=True,
		return_tensors="pt"
	)
}, input_columns="query", num_proc=16)

# Sentiment Analysis
analyser = pipeline(
	model=wandb.config["sentiment_analysis_model"],
 	tokenizer=wandb.config["sentiment_analysis_model"],
	max_length=512,
	truncation=True,
	framework="pt",
	task="sentiment-analysis",
	num_workers=16,
	device_map="auto",
	torch_dtype="auto",
	model_kwargs={
		"quantization_config": BitsAndBytesConfig(
			load_in_4bit=True,
			bnb_4bit_compute_dtype=torch.float16
		),
		"id2label": {k: v for k, v in enumerate(emotion_labels)},
		"label2id": {v: k for k, v in enumerate(emotion_labels)},
		"low_cpu_mem_usage": True
	},
	trust_remote_code=True
)

sentiment_analysis_model = torch.compile(analyser.model)


# [TODO] a reward function contain length and emotion
target_length = 69
# the length of output that we prefer

def calculate_emotion_score(response: str, correct_emotion: str) -> float:
    # correct: save the score from analyser 
    # wrong: [TO-DO] (save 1 - score from analyser )
    emotion_output = analyser(response)[0]
    print(emotion_output)
    if emotion_output["label"] == correct_emotion:
        emotion_score = emotion_output["score"] * 10
    else:
        emotion_score = 1 - emotion_output["score"]
    return emotion_score

def calculate_length_score(response_length: int) -> float:
    # use reciprocal of length difference to calculate
    # the larger the difference the smaller the score is
    length_diff = abs(response_length - target_length)
    print("len and len diff",response_length, length_diff)
    length_score = 1 / (length_diff + 1)
    return length_score

def reward(batch: dict) -> list:
    print("Hello Huston, here is a reward function")
    rewards = []
    res_len = []
    for response, response_length, raw_correct_emotion in zip(batch["response"], batch["response_length"], batch["label"]):
        correct_emotion = emotion_labels[raw_correct_emotion]
        res_len.append(len(response))
        
        emotion_score = calculate_emotion_score(response, correct_emotion)
        length_score = calculate_length_score(response_length)
        # use the product of two score as reward
        reward_product = emotion_score * length_score
        rewards.append(reward_product)
    print("\ntarget length: ", target_length)
    print("response length:")
    import statistics
    print("max:", max(res_len),"\nmin:", min(res_len),"\navg:", statistics.mean(res_len))
    
    return rewards

ppo_config = PPOConfig(
	gradient_accumulation_steps=1,
	learning_rate=wandb.config["learning_rate"],
	max_grad_norm=wandb.config["max_grad_norm"],
	log_with="wandb",
	optimize_device_cache=True,
	early_stopping=True,
	is_peft_model=True,
	use_score_scaling=True,
	use_score_norm=True,
	score_clip=wandb.config["score_clip"],
)

optimizer = PagedLion32bit(filter(lambda p: p.requires_grad, base_model.parameters()), lr=ppo_config.learning_rate)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=wandb.config["lr_gamma"])
length_sampler = LengthSampler(wandb.config["min_new_tokens"], wandb.config["max_new_tokens"])

streamer = TextStreamer(
	tokenizer,
	skip_special_tokens=True, # show <pad> or not
	clean_up_tokenization_spaces=True
)

generation_config = GenerationConfig(
	max_length=None,
	min_length=-1,
	top_k=wandb.config["top_k"],
	top_p=wandb.config["top_p"],
	do_sample=True,
	use_cache=True,
	repetition_penalty=wandb.config["repetition_penalty"],
	pad_token_id=tokenizer.pad_token_id,
	bos_token_id=tokenizer.bos_token_id,
	eos_token_id=tokenizer.eos_token_id,
	low_memory=True
)

# Setup Tuner
tuner = PPOTrainer(
	config=ppo_config,
	model=ppo_model,
	tokenizer=tokenizer,
	dataset=dataset,
	optimizer=optimizer,
	lr_scheduler=lr_scheduler
)

for epoch in range(wandb.config["num_epoches"]):
	for batch in tqdm(tuner.dataloader, desc=f"epoch{epoch}", colour="yellow"):
<<<<<<< HEAD
		query_tensors = batch["input_ids"]
=======
		query_tensors = [input_ids.squeeze(0) for input_ids in batch["input_ids"]]

>>>>>>> main
		response_tensors = tuner.generate(
			query_tensors,
			return_prompt=False,
			batch_size=1,   # must set to 1 if using streamer
			streamer=streamer,  # use streamer to show the generation process
			length_sampler=length_sampler,
			**generation_config.to_dict()
		)
		batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
		# [TODO] batch["response_length"] 
		batch["response_length"] = [r.size()[0] for r in response_tensors] 
		response_tensors = [torch.LongTensor(t.to("cpu")) for t in response_tensors]

		reward_scores = reward(batch)
		rewards = [torch.FloatTensor(torch.tensor(scores, device="cpu")) for scores in reward_scores]

		stats = tuner.step(query_tensors, response_tensors, rewards)
		tuner.log_stats(stats, batch, rewards)

tuner.model = torch.compile(tuner.model)
tuner.model.push_to_hub(repo_id="response_generator_for_emotion_chat_bot", commit="", create_pr=True)

wandb.finish()
