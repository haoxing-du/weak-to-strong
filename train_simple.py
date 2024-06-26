import json
import os
import random
import subprocess
from typing import Dict, List, Optional

import fire
import numpy as np
import torch
from datasets import load_dataset, load_from_disk, concatenate_datasets

import weak_to_strong.logger as logger
from weak_to_strong.common import get_tokenizer
from weak_to_strong.datasets import (VALID_DATASETS, load_dataset,
                                     tokenize_dataset)
from weak_to_strong.loss import logconf_loss_fn, product_loss_fn, xent_loss
from weak_to_strong.train import ModelConfig, train_and_save_model

# NOTE learning rates are not particularly tuned, work somewhat reasonably at train batch size 32
MODEL_CONFIGS = [
    ModelConfig(
        name="gpt2",
        default_lr=5e-5,
        eval_batch_size=32,
    ),
    ModelConfig(
        name="gpt2-medium",
        default_lr=5e-5,
        eval_batch_size=32,
    ),
    ModelConfig(
        name="gpt2-large",
        default_lr=1e-5,
        eval_batch_size=32,
    ),
    ModelConfig(
        name="gpt2-xl",
        default_lr=1e-5,
        eval_batch_size=2,
        gradient_checkpointing=True,
        # Should use model_parallel on V100s (note: ironically if you have a single V100 it should run,
        # but if you have multiple it won't run without model_parallel because of the overhead of data
        # parallel training).
        model_parallel=(
            torch.cuda.get_device_properties(0).total_memory < 35e9
            and torch.cuda.device_count() > 1
        ),
    ),
    ModelConfig(
        name="Qwen/Qwen-1_8B",
        default_lr=1e-5,
        eval_batch_size=2,
        gradient_checkpointing=True,
        model_parallel=(
            torch.cuda.get_device_properties(0).total_memory < 35e9
            and torch.cuda.device_count() > 1
        ),
        custom_kwargs={
            "trust_remote_code": True,
            "bf16": torch.cuda.is_bf16_supported(),
            "fp32": not torch.cuda.is_bf16_supported(),
            "revision": "5fde88dff770a7d036847211f5d9d9705f0caa69",
        },
    ),
    ModelConfig(
        name="Qwen/Qwen-7B",
        default_lr=1e-5,
        eval_batch_size=2,
        gradient_checkpointing=True,
        model_parallel=True,
        # note: you will probably not be able to run this without many gpus
        custom_kwargs={
            "trust_remote_code": True,
            "bf16": torch.cuda.is_bf16_supported(),
            "fp32": not torch.cuda.is_bf16_supported(),
            "revision": "d4efd21e866b9cb3466cb65b963933f5e98016d1",
        },
    ),
    ModelConfig(
        name="Qwen/Qwen-14B",
        default_lr=1e-5,
        eval_batch_size=2,
        gradient_checkpointing=True,
        model_parallel=True,
        # note: you will probably not be able to run this bf16 support and without many gpus
        custom_kwargs={
            "trust_remote_code": True,
            "bf16": torch.cuda.is_bf16_supported(),
            "fp32": not torch.cuda.is_bf16_supported(),
            "revision": "8be2854218fea9054331e217fd26a06f3fd02004",
        },
    ),
    ModelConfig(
        name="Qwen/Qwen-72B",
        default_lr=1e-5,
        eval_batch_size=1,
        gradient_checkpointing=True,
        model_parallel=True,
        # note: you will probably not be able to run this without bf16 support and many gpus
        custom_kwargs={
            "trust_remote_code": True,
            "bf16": torch.cuda.is_bf16_supported(),
            "fp32": not torch.cuda.is_bf16_supported(),
            "revision": "fec78c0e3b3b10dd9f0ce775c34a686a3255a7d1",
        },
        # This model is really big, save space by using adafactor.
        # Note that even then it will take up ~60GB per GPU on an 8-GPU machine.
        default_optimizer="adafactor",
    ),
]
MODELS_DICT: Dict[str, ModelConfig] = {
    model_config.name: model_config for model_config in MODEL_CONFIGS
}


loss_dict = {
    "logconf": logconf_loss_fn(),
    "product": product_loss_fn(),
    "xent": xent_loss(),
}

VALID_LOSSES: List[str] = list(loss_dict.keys())


def get_config_foldername(config: dict) -> str:
    def shorten_key(key: str) -> str:
        return "".join(word[0] for word in key.split("_"))

    def shorten_value(value) -> str:
        if isinstance(value, bool):
            return "1" if value else "0"
        elif isinstance(value, str):
            value = value.split("/")[-1]
            if "_" in value:
                return "_".join(word[:4] for word in value.split("_"))
            else:
                return value
        else:
            return str(value)

    return "-".join(f"{shorten_key(k)}={shorten_value(v)}" for k, v in sorted(config.items()))


def main(
    batch_size: int = 32,
    max_ctx: int = 1024,
    ds_name: str = "sciq",
    loss: str = "xent",
    n_docs: int = 20000,
    n_test_docs: int = 10000,
    model_size: str = "gpt2",
    lr: Optional[float] = None,
    optim: Optional[str] = None,
    epochs: int = 2,
    force_retrain: bool = False,
    seed: int = 0,
    minibatch_size_per_device: Optional[float] = None,
    train_with_dropout: bool = False,
    results_folder: str = "/tmp/results",
    linear_probe: bool = False,
    lr_schedule: str = "cosine_anneal",
    # Note: you can pass either weak_model_size or weak_labels_path. If you pass
    # weak_model_size, we will guess the path to the weak labels based on the weak
    # model. If you pass weak_labels_path, we will use that path instead.
    # If you pass neither, we will train on ground truth.
    weak_model_size: Optional[str] = None,
    weak_labels_path: Optional[str] = None,
    strong_label_fraction: Optional[float] = 0,
    shuffle_strong_labels: Optional[bool] = True,
    strong_after_weak: Optional[bool] = False,
    replace_least_confident: Optional[bool] = False,
    replace_most_incorrect: Optional[bool] = False,
    sweep_subfolder: str = "default",
    # Set to a very large value so that by default we don't do any intermediate evals but
    # still do final evals (which requires eval_every to be set to a non-zero, non-None value)
    eval_every: int = 1000000,
    sync_command: Optional[str] = None,
):
    # this is per device!
    if minibatch_size_per_device is None:
        minibatch_size_per_device = 1
    if model_size == 'gpt2-xl':
        minibatch_size_per_device = 1
        print("Changing minibatch_size_per_device to 1 for gpt2-xl")
    assert ds_name in VALID_DATASETS, f"Unknown dataset {ds_name} not in {VALID_DATASETS}"
    assert (
        weak_model_size is None or weak_labels_path is None
    ), "Can't pass both weak_model_size and weak_labels_path"
    assert strong_label_fraction >= 0 and strong_label_fraction <= 1, "Invalid strong_label_fraction"
    model_config = MODELS_DICT[model_size]
    assert not (shuffle_strong_labels and strong_after_weak), "Can't both shuffle and place strong labels after weak"
    assert not (replace_least_confident and replace_most_incorrect), "Can't pass both replace_least_confident and replace_most_incorrect"

    use_default_lr = False
    if lr is None:
        assert (
            batch_size == 32
        ), "Learning rates were tuned on batch size 32, you probably want to sweep LR if you are tuning batch size"
        lr = model_config.default_lr
        use_default_lr = True

    if optim is None:
        optim = model_config.default_optimizer

    # The commented out terms are the ones that should not change final results
    config = {
        "batch_size": batch_size,
        "max_ctx": max_ctx,
        "ds_name": ds_name,
        "loss": loss,
        "n_docs": n_docs,
        "n_test_docs": n_test_docs,
        "model_size": model_size,
        "lr": lr,
        "optim": optim,
        "epochs": epochs,
        # "force_retrain": force_retrain,
        "seed": seed,
        # "minibatch_size_per_device": minibatch_size_per_device,
        "train_with_dropout": train_with_dropout,
        # "results_folder": results_folder,
        "linear_probe": linear_probe,
        "lr_schedule": lr_schedule,
        "eval_every": eval_every,
        # "sweep_subfolder": sweep_subfolder,
    }

    if weak_model_size is not None:
        weak_model_config = config.copy()
        weak_model_config["model_size"] = weak_model_size
        weak_model_config["loss"] = "xent"
        if use_default_lr:
            weak_model_config["lr"] = MODELS_DICT[weak_model_size].default_lr

        weak_model_config_name = get_config_foldername(weak_model_config)

        weak_labels_path = (
            results_folder + "/" + sweep_subfolder + "/" + weak_model_config_name + "/weak_labels"
        )

    eval_batch_size = model_config.eval_batch_size
    random.seed(seed)

    # Load dataset
    dataset = load_dataset(ds_name, seed=seed, split_sizes=dict(train=n_docs, test=n_test_docs))

    # Split the training dataset in half
    train_dataset, test_ds = dataset["train"], dataset["test"]

    if weak_labels_path is None:
        split_data = train_dataset.train_test_split(test_size=0.5, seed=seed)
        train1_ds, train2_ds = split_data["train"], split_data["test"]
        print("len(train1):", len(train1_ds), "len(train2):", len(train2_ds))
        config_name = get_config_foldername(config)
    else:
        if not weak_labels_path.endswith("weak_labels"):
            weak_labels_path = weak_labels_path + "/weak_labels"
        if sync_command is not None:
            sync_command_list = sync_command.split(" ")
            sync_command_list.extend(
                ["download", weak_labels_path.replace("/weak_labels", ""), results_folder]
            )
            print(f"Running sync command: {' '.join(sync_command_list)}")
            result = subprocess.run(sync_command_list, check=True)
            if result.returncode != 0:
                raise RuntimeError(f"Sync command failed with return code {result.returncode}")
        print("Weak labels path:", weak_labels_path)
        train1_ds = load_from_disk(weak_labels_path)
        train2_ds = None

        weak_model_config = json.load(open(weak_labels_path.replace("weak_labels", "config_w2sg.json")))
        # mix in strong labels at the desired fraction
        if strong_label_fraction > 0:
            n_strong, n_weak = int(len(train1_ds) * strong_label_fraction), int(len(train1_ds) * (1 - strong_label_fraction))
            # strong labels are the ground truth labels
            split_data = train_dataset.train_test_split(test_size=0.5, seed=seed)
            first_half, _ = split_data["train"], split_data["test"]
            strong_labels = first_half.select(range(n_strong))
            if replace_most_incorrect:
                def disagreement_score(example):
                    soft_label = np.array(example['soft_label'])
                    hard_label = np.array([1, 0] if example['hard_label'] == 0 else [0, 1])
                    disagreement = np.sum(np.abs(soft_label - hard_label))
                    return {'disagreement': disagreement}
                train1_ds = train1_ds.map(disagreement_score, batched=False)
                train1_ds = train1_ds.sort('disagreement', reverse=False) # sort from least disagreement to most
                weak_labels = train1_ds.select(range(n_weak)) # select the least disagreement examples
            elif replace_least_confident:
                def confidence_score(example):
                    soft_label = np.array(example['soft_label'])
                    confidence = np.max(soft_label)
                    return {'confidence': confidence}
                train1_ds = train1_ds.map(confidence_score, batched=False)
                train1_ds = train1_ds.sort('confidence', reverse=True) # sort from most confident to least
                weak_labels = train1_ds.select(range(n_weak)) # select the most confident examples
            else:
                weak_labels = train1_ds.select(range(n_weak))
            print("len(strong_labels):", len(strong_labels), "len(weak_labels):", len(weak_labels))
            if strong_after_weak:
                train1_ds = concatenate_datasets([weak_labels, strong_labels])
            else:
                train1_ds = concatenate_datasets([strong_labels, weak_labels])
            if shuffle_strong_labels:
                print("Shuffling strong labels")
                train1_ds = train1_ds.shuffle(seed=seed)
            else:
                print("Not shuffling strong labels")
        # print first 10 elements of the dataset
        print("train1_ds[:10] ", train1_ds[:10])
        config["weak_model_size"] = weak_model_config["model_size"]
        config["strong_label_fraction"] = strong_label_fraction
        config["shuffle_strong_labels"] = shuffle_strong_labels
        if strong_after_weak:
            config["strong_after_weak"] = strong_after_weak
        if replace_least_confident:
            config["replace_least_confident"] = replace_least_confident
        if replace_most_incorrect:
            config["replace_most_incorrect"] = replace_most_incorrect
        config_name = get_config_foldername(config)
        config["weak_model"] = weak_model_config

    save_path = os.path.join(results_folder, sweep_subfolder, config_name)
    logger.configure(
        name="{sweep_subfolder}_{config_name}_{datetime_now}",
        save_path=save_path,
        sweep_subfolder=sweep_subfolder,
        config_name=config_name,
    )
    # Tokenize datasets
    tokenizer = get_tokenizer(model_config.name)
    train1_ds = tokenize_dataset(train1_ds, tokenizer, max_ctx)
    test_ds = tokenize_dataset(test_ds, tokenizer, max_ctx)
    if train2_ds:
        train2_ds = tokenize_dataset(train2_ds, tokenizer, max_ctx)

    loss_fn = loss_dict[loss]
    print(f"Training model model, size {model_size}")
    test_results, weak_ds = train_and_save_model(
        model_config,
        train1_ds,
        test_ds,
        inference_ds=train2_ds,
        batch_size=batch_size,
        save_path=save_path,
        loss_fn=loss_fn,
        lr=lr,
        epochs=epochs,
        force_retrain=force_retrain,
        eval_batch_size=eval_batch_size,
        minibatch_size_per_device=minibatch_size_per_device,
        train_with_dropout=train_with_dropout,
        linear_probe=linear_probe,
        lr_schedule=lr_schedule,
        optimizer_name=optim,
        eval_every=eval_every,
    )

    if weak_ds is not None:
        weak_ds.save_to_disk(save_path + "/" + "weak_labels")

    accs = [x["acc"] for x in test_results]
    acc = np.mean(accs)
    stderr = np.std(accs) / np.sqrt(len(accs))
    res_dict = {"accuracy": acc, "stderr": stderr}
    print("accuracy:", acc, "+/-", stderr)

    if not os.path.exists(os.path.join(save_path, "config_w2sg.json")):
        with open(os.path.join(save_path, f"config_w2sg.json"), "w") as f:
            json.dump(config, f, indent=2)

    if not os.path.exists(os.path.join(save_path, "results_summary.json")):
        with open(os.path.join(save_path, f"results_summary.json"), "w") as f:
            json.dump(res_dict, f, indent=2)

    if sync_command is not None:
        print("Syncing results to remote storage...")
        try:
            sync_command_list = sync_command.split(" ")
            sync_command_list.extend(["upload", save_path, results_folder])
            print(f"Running sync command: {' '.join(sync_command_list)}")
            result = subprocess.run(sync_command_list, check=True)
            if result.returncode != 0:
                raise RuntimeError(f"Sync command failed with return code {result.returncode}")
        except Exception as e:
            raise RuntimeError("Failed to sync results to remote storage.") from e


if __name__ == "__main__":
    fire.Fire(main)
