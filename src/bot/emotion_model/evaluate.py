from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Optional

import torch
from datasets import load_from_disk
from safetensors.torch import load_model
from torcheval.metrics.functional import multiclass_f1_score, multiclass_accuracy
from transformers.hf_argparser import HfArgumentParser, HfArg

import wandb

from libs import EmotionModel, representation_evolute, CommonScriptArguments, CommonWanDBArguments, get_torch_device


@dataclass
class ScriptArguments(CommonScriptArguments):
    dtype: Optional[str] = HfArg(aliases="--dtype", default="torch.float32")
    device: Optional[str] = HfArg(aliases="--device", default_factory=get_torch_device)


config_getter = ArgumentParser()
config_getter.add_argument("--json_file", required=True, type=str)
config = config_getter.parse_args()

parser = HfArgumentParser((ScriptArguments, CommonWanDBArguments))
args, wandb_args = parser.parse_json_file(config.json_file)
dtype: torch.dtype = eval(args.dtype)

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

dataset_path = run.use_artifact(wandb.config["dataset"]).download()
eval_dataset = load_from_disk(dataset_path)["test"]

model_path = run.use_artifact(wandb.config["model"], type="model").download()
model = torch.compile(EmotionModel(dtype=dtype, device=args.device))
load_model(model, f"{model_path}/emotion_model.safetensors")

eval_dataset = eval_dataset.map(lambda samples: {
    "bot_representation": [representation_evolute(model, sample[0], sample[1])
                           for sample in zip(samples["bot_representation"],
                                             samples["user_dialog_emotion_composition"])]
}, batched=True)

eval_dataset = eval_dataset.map(lambda samples: {
    "bot_most_possible_emotion": [torch.argmax(torch.tensor(sample), dim=1) for sample in samples]
}, input_columns="bot_representation", batched=True, num_proc=16)

predicted_labels: torch.tensor = torch.cat([torch.tensor(turn) for turn in eval_dataset["bot_most_possible_emotion"]])
true_labels: torch.tensor = torch.cat([torch.tensor(turn) for turn in eval_dataset["bot_emotion"]])

wandb.log({
    "F1-score": multiclass_f1_score(true_labels, predicted_labels, num_classes=7, average="weighted"),
    "Accuracy": multiclass_accuracy(true_labels, predicted_labels, num_classes=7)
})

emotion_labels: list = ["neutral", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]
eval_dataset = eval_dataset.map(lambda samples: {
    "bot_most_possible_emotion": [[emotion_labels[emotion_id] for emotion_id in sample]
                                  for sample in samples["bot_most_possible_emotion"]],
    "bot_emotion": [[emotion_labels[emotion_id] for emotion_id in sample] for sample in samples["bot_emotion"]]
}, batched=True, num_proc=16)

result = eval_dataset.map(lambda samples: {
    "bot_most_possible_emotion": [", ".join(sample) for sample in samples["bot_most_possible_emotion"]],
    "bot_emotion": [", ".join(sample) for sample in samples["bot_emotion"]]
}, batched=True, num_proc=16)

result = result.remove_columns(["bot_representation", "bot_dialog", "user_dialog", "user_dialog_emotion_composition"])

wandb.log({"evaluation_result": wandb.Table(dataframe=result.to_pandas())})

wandb.finish()
