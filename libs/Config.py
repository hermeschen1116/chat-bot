from dataclasses import dataclass
from typing import Any, Optional, Union, Dict, Sequence, List

from wandb.sdk import Settings
from wandb.sdk.lib.paths import StrPath


def value_candidate_check(input_value: Any,
                          possible_values: list,
                          use_default_value: bool,
                          default_value: Optional[Any]) -> Any:
    if input_value not in possible_values:
        error_message: str = f"{input_value} should be any of {','.join(possible_values)}"
        if use_default_value:
            print(error_message)
            return default_value
        raise ValueError(error_message)

    return input_value


@dataclass
class WanDBArguments:
    job_type: Optional[str] = None,
    dir: Optional[StrPath] = None,
    config: Union[Dict, str, None] = None,
    project: Optional[str] = None,
    entity: Optional[str] = None,
    reinit: Optional[bool] = None,
    tags: Optional[Sequence] = None,
    group: Optional[str] = None,
    name: Optional[str] = None,
    notes: Optional[str] = None,
    magic: Optional[Union[dict, str, bool]] = None,
    config_exclude_keys: Optional[List[str]] = None,
    config_include_keys: Optional[List[str]] = None,
    anonymous: Optional[str] = None,
    mode: Optional[str] = None,
    allow_val_change: Optional[bool] = None,
    resume: Optional[Union[bool, str]] = None,
    force: Optional[bool] = None,
    tensorboard: Optional[bool] = None,
    sync_tensorboard: Optional[bool] = None,
    monitor_gym: Optional[bool] = None,
    save_code: Optional[bool] = None,
    id: Optional[str] = None,
    fork_from: Optional[str] = None,
    settings: Union[Settings, Dict[str, Any], None] = None

    def __post_init__(self):
        module: list = ["sentiment_analysis",
                        "candidate_generator",
                        "emotion_predictor",
                        "emotion_model",
                        "similarity_analysis",
                        "response_generator"]

        wandb_mode: list = ["online", "offline", "disabled"]

        wandb_resume_mode: list = ["allow", "must", "never", "auto"]

        self.group = value_candidate_check(self.group,
                                           use_default_value=True,
                                           default_value="",
                                           possible_values=module)
        self.mode = value_candidate_check(self.mode,
                                          use_default_value=True,
                                          default_value="online",
                                          possible_values=wandb_mode)
        self.resume = value_candidate_check(self.resume,
                                            use_default_value=True,
                                            default_value=None,
                                            possible_values=wandb_resume_mode)


@dataclass
class TrainerArguments:
    output_dir: str
    overwrite_output_dir: bool = False
    do_train: bool = False
    do_eval: bool = False
    do_predict: bool = False
    evaluation_strategy: Union = 'no'
    prediction_loss_only: bool = False
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    per_gpu_train_batch_size: Optional = None
    per_gpu_eval_batch_size: Optional = None
    gradient_accumulation_steps: int = 1
    eval_accumulation_steps: Optional = None
    eval_delay: Optional = 0
    earning_rate: float = 5e-05
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-08
    max_grad_norm: float = 1.0
    num_train_epochs: float = 3.0
    max_steps: int = -1
    r_scheduler_type: Union = 'linear'
    lr_scheduler_kwargs: Optional = dict
    warmup_ratio: float = 0.0
    warmup_steps: int = 0
    og_level: Optional = 'passive'
    log_level_replica: Optional = 'warning'
    log_on_each_node: bool = True
    logging_dir: Optional = None
    logging_strategy: Union = 'steps'
    logging_first_step: bool = False
    logging_steps: float = 500
    ogging_nan_inf_filter: bool = True
    save_strategy: Union = 'steps'
    save_steps: float = 500
    save_total_limit: Optional = None
    save_safetensors: Optional = True
    save_on_each_node: bool = False
    save_only_model: bool = False
    no_cuda: bool = False
    use_cpu: bool = False
    use_mps_device: bool = False
    seed: int = 42
    data_seed: Optional = None
    jit_mode_eval: bool = False
    use_ipex: bool = False
    bf16: bool = False
    fp16: bool = False
    fp16_opt_level: str = 'O1'
    half_precision_backend: str = 'auto'
    bf16_full_eval: bool = False
    fp16_full_eval: bool = False
    tf32: Optional = None
    local_rank: int = -1
    ddp_backend: Optional = None
    tpu_num_cores: Optional = None
    tpu_metrics_debug: bool = False
    debug: Union = ''
    dataloader_drop_last: bool = False
    eval_steps: Optional = None
    dataloader_num_workers: int = 0
    dataloader_prefetch_factor: Optional = None
    past_index: int = -1
    run_name: Optional = None
    disable_tqdm: Optional = None
    remove_unused_columns: Optional = True
    label_names: Optional = None
    load_best_model_at_end: Optional = False
    metric_for_best_model: Optional = None
    greater_is_better: Optional = None
    ignore_data_skip: bool = False
    fsdp: Union = ''
    fsdp_min_num_params: int = 0
    fsdp_config: Union = None
    fsdp_transformer_layer_cls_to_wrap: Optional = None
    accelerator_config: Optional = None
    deepspeed: Optional = None
    label_smoothing_factor: float = 0.0
    optim: Union = 'adamw_torch'
    optim_args: Optional = None
    adafactor: bool = False
    group_by_length: bool = False
    length_column_name: Optional = 'length'
    report_to: Optional = None
    ddp_find_unused_parameters: Optional = None
    ddp_bucket_cap_mb: Optional = None
    ddp_broadcast_buffers: Optional = None
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = False
    skip_memory_metrics: bool = True
    use_legacy_prediction_loop: bool = False
    push_to_hub: bool = False
    resume_from_checkpoint: Optional = None
    hub_model_id: Optional = None
    hub_strategy: Union = 'every_save'
    hub_token: Optional = None
    hub_private_repo: bool = False
    hub_always_push: bool = False
    gradient_checkpointing: bool = False
    gradient_checkpointing_kwargs: Optional = None
    include_inputs_for_metrics: bool = False
    fp16_backend: str = 'auto'
    push_to_hub_model_id: Optional = None
    push_to_hub_organization: Optional = None
    push_to_hub_token: Optional = None
    mp_parameters: str = ''
    auto_find_batch_size: bool = False
    full_determinism: bool = False
    torchdynamo: Optional = None
    ray_scope: Optional = 'last'
    ddp_timeout: Optional = 1800
    torch_compile: bool = False
    torch_compile_backend: Optional = None
    torch_compile_mode: Optional = None
    dispatch_batches: Optional = None
    split_batches: Optional = None
    include_tokens_per_second: Optional = False
    include_num_input_tokens_seen: Optional = False
    neftune_noise_alpha: Optional = None
    optim_target_modules: Union = None