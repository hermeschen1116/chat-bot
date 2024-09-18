import json
from argparse import ArgumentParser

import torch
import torch.nn.functional as f
import wandb
from datasets import load_dataset
from libs import EmotionModel, calculate_evaluation_result, get_torch_device, login_to_service, representation_evolute
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def sweep_function(config: dict = None) -> None:
	run = wandb.init(job_type="Sweep", project="emotion-chat-bot-ncu", group="Emotion Model", config=config)
	device: str = get_torch_device()
	dtype = eval(run.config["dtype"])

	# Load Dataset
	dataset = load_dataset(
		"hermeschen1116/emotion_transition_from_dialog",
		num_proc=16,
		trust_remote_code=True,
	)

	model = EmotionModel(
		attention=run.config["attention"],
		dropout=run.config["dropout"],
		bias=run.config["bias"],
		dtype=dtype,
		device=device,
	)

	loss_function = torch.nn.CrossEntropyLoss()
	optimizer = eval(f"torch.optim.{run.config['optimizer']}")(model.parameters(), lr=run.config["learning_rate"])

	train_dataloader = DataLoader(dataset["train"])
	validation_dataloader = DataLoader(dataset["validation"])
	for i in range(run.config["num_epochs"]):
		running_loss: float = 0
		model.train()
		for sample in tqdm(train_dataloader, colour="green"):
			representation, emotion_composition = (
				sample["bot_representation"],
				sample["user_emotion_composition"],
			)
			labels = f.one_hot(torch.cat(sample["bot_emotion"]), 7).to(dtype)

			optimizer.zero_grad()

			output = representation_evolute(model, representation, emotion_composition)

			loss = loss_function(output, labels)
			wandb.log({"train/loss": loss.item()})
			running_loss += loss.item()

			loss.backward()
			optimizer.step()

		if i + 1 == run.config["num_epochs"]:
			wandb.log({"train/train_loss": running_loss / len(train_dataloader)})

		running_loss = 0
		truths: list = []
		predictions: list = []
		model.eval()
		with torch.no_grad():
			for sample in tqdm(validation_dataloader, colour="blue"):
				representation, emotion_composition = (
					sample["bot_representation"],
					sample["user_emotion_composition"],
				)
				labels = f.one_hot(torch.cat(sample["bot_emotion"]), 7).to(dtype)

				output = representation_evolute(model, representation, emotion_composition)

				loss = loss_function(output, labels)
				wandb.log({"val/loss": loss.item()})
				running_loss += loss.item()
				truths += sample["bot_emotion"]
				predictions.append(torch.argmax(output, dim=1))

			wandb.log({"val/loss": running_loss / len(validation_dataloader)})

			evaluation_result: dict = calculate_evaluation_result(torch.cat(predictions), torch.cat(truths))
			wandb.log({"val/f1_score": evaluation_result["f1_score"], "val/accuracy": evaluation_result["accuracy"]})

	model.eval()
	model = torch.compile(model)

	eval_dataset = dataset["test"].map(
		lambda samples: {
			"bot_representation": [
				representation_evolute(model, sample[0], sample[1])
				for sample in zip(
					samples["bot_representation"],
					samples["user_emotion_composition"],
				)
			]
		},
		batched=True,
	)

	eval_dataset = eval_dataset.map(
		lambda samples: {
			"bot_most_possible_emotion": [torch.argmax(torch.tensor(sample), dim=1) for sample in samples]
		},
		input_columns="bot_representation",
		batched=True,
		num_proc=16,
	)

	eval_predictions: Tensor = torch.cat([torch.tensor(turn) for turn in eval_dataset["bot_most_possible_emotion"]])
	eval_truths: Tensor = torch.cat([torch.tensor(turn) for turn in eval_dataset["bot_emotion"]])

	evaluation_result: dict = calculate_evaluation_result(eval_predictions, eval_truths)
	wandb.log(
		{
			"eval/f1-score": evaluation_result["f1_score"],
			"eval/accuracy": evaluation_result["accuracy"],
			"eval/optimize_metric": torch.tensor(list(evaluation_result.values())).dot(torch.tensor([0.5, 0.5])),
		}
	)


login_to_service()
config_getter = ArgumentParser()
config_getter.add_argument("--json_file", required=True, type=str)
config = config_getter.parse_args()

with open(config.json_file, "r", encoding="utf-8") as config_file:
	sweep_config: dict = json.load(config_file)

sweep_id = wandb.sweep(sweep=sweep_config, project="emotion-chat-bot-ncu")
wandb.agent(sweep_id, sweep_function, project="emotion-chat-bot-ncu", count=100)
wandb.finish()