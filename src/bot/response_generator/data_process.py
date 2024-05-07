import tempfile
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Optional

import wandb
from datasets import load_dataset
from transformers import HfArgumentParser
from transformers.hf_argparser import HfArg

from libs.CommonConfig import CommonScriptArguments, CommonWanDBArguments


@dataclass
class ScriptArguments(CommonScriptArguments):
    dataset_name: Optional[str] = HfArg(aliases="--dataset-name", default="daily_dialog_for_RG")


config_getter = ArgumentParser()
config_getter.add_argument("--json_file", required=True, type=str)
config = config_getter.parse_args()

parser = HfArgumentParser((ScriptArguments, CommonWanDBArguments))
args, wandb_args = parser.parse_json_file(config.json_file)

run = wandb.init(
    job_type=wandb_args.job_type,
    config=wandb_args.config,
    project=wandb_args.project,
    group=wandb_args.group,
    notes=wandb_args.notes,
    mode=wandb_args.mode,
    resume=wandb_args.resume
)

dataset = load_dataset("daily_dialog",
                       num_proc=16, save_infos=True,
                       trust_remote_code=True).remove_columns("act")

dataset = dataset.rename_column("emotion", "emotion_id")
emotion_labels: list = dataset["train"].features["emotion_id"].feature.names
emotion_labels[0] = "neutral"
dataset = dataset.map(
    lambda samples: {"emotion": [[emotion_labels[emotion_id] for emotion_id in sample] for sample in samples]},
    input_columns="emotion_id", remove_columns="emotion_id", batched=True, num_proc=16)

dataset = dataset.map(lambda samples: {"dialog": [[dialog.strip() for dialog in sample] for sample in samples]},
                      input_columns="dialog", batched=True, num_proc=16)

dataset = dataset.map(lambda samples: {
    "emotion": [sample[:-1] if len(sample) % 2 == 1 else sample for sample in samples["emotion"]],
    "dialog": [sample[:-1] if len(sample) % 2 == 1 else sample for sample in samples["dialog"]]
}, batched=True, num_proc=16)

dataset = dataset.map(lambda samples: {
    "emotion": [sample for sample in samples["emotion"] if len(sample) != 0],
    "dialog": [sample for sample in samples["dialog"] if len(sample) != 0]
}, batched=True, num_proc=16)

dataset["test"] = dataset["test"].map(lambda samples: {
    "emotion_history": [sample[:-1] for sample in samples],
    "emotion_bot": [sample[-1] for sample in samples]
}, input_columns="emotion", batched=True, num_proc=16)

dataset["test"] = dataset["test"].map(lambda samples: {
    "dialog_history": [sample[:-1] for sample in samples],
    "dialog_bot": [sample[-1] for sample in samples]
}, input_columns="dialog", batched=True, num_proc=16)

dataset = dataset.map(lambda samples: {
    "prompt": [[{
        "role": "user" if i % 2 == 0 else "assistant",
        "content": {"emotion": emotion, "dialog": dialog}}
        for i, (emotion, dialog) in enumerate(zip(sample[0], sample[1]))]
        for sample in zip(samples["emotion"], samples["dialog"])]
}, remove_columns=["emotion", "dialog"], batched=True, num_proc=16)

dataset["test"] = dataset["test"].map(lambda samples: {
    "prompt": [[{
        "role": "user" if i % 2 == 0 else "assistant",
        "content": {"emotion": emotion, "dialog": dialog}}
        for i, (emotion, dialog) in enumerate(zip(sample[0], sample[1]))]
        for sample in zip(samples["emotion_history"], samples["dialog_history"])]
}, batched=True, num_proc=16)

dataset["test"] = dataset["test"].map(lambda sample: {
    "history": ["\n".join([f"""{'user' if i % 2 == 0 else 'bot'}({v[0]}): {v[1]}"""
                           for i, v in enumerate(zip(sample["emotion_history"], sample["dialog_history"]))])]
}, remove_columns=["emotion_history", "dialog_history"], num_proc=16)

test_dataset_artifact = wandb.Artifact(
    f"{args.dataset_name}_test",
    type="dataset",
    description="modified version of daily dialog dataset from huggingface for response generator module"
)

with tempfile.TemporaryDirectory() as temp_dir:
    dataset["test"].save_to_disk(f"{temp_dir}/{args.dataset_name}_test", num_proc=16)
    test_dataset_artifact.add_dir(f"{temp_dir}/{args.dataset_name}_test")
    run.log_artifact(test_dataset_artifact)

dataset_artifact = wandb.Artifact(
    f"{args.dataset_name}_train",
    type="dataset",
    description="modified version of daily dialog dataset from huggingface for response generator module"
)

with tempfile.TemporaryDirectory() as temp_dir:
    dataset["train"].save_to_disk(f"{temp_dir}/{args.dataset_name}_train", num_proc=16)
    dataset_artifact.add_dir(f"{temp_dir}/{args.dataset_name}_train")
    run.log_artifact(dataset_artifact)

wandb.finish()