import sys
sys.path.append('.')
import yaml
import os
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from core.dataset import collate_fn
from core.PC_NET import PCNet
from core.config import punct_label2id, cap_label2id, MODEL_ID

# Optional performance tuning
torch.set_float32_matmul_precision('high')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch._dynamo.config.cache_size_limit = 4 * 2048


def load_yaml_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def create_dynamic_log_dir(config):
    model_name = "PCNet"
    lr = config['optimizer']['learning_rate']
    layers_to_train = int(config['layers_to_train'])
    log_dir = f"{config['logging']['log_dir_base']}/{model_name}_lr{lr}_layers{layers_to_train}"
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def setup_scheduler(scheduler_config):
    if not scheduler_config.get("enabled", False):
        return None

    scheduler_type = scheduler_config["type"]
    params = scheduler_config["params"]

    if scheduler_type == "ReduceLROnPlateau":
        return {
            "type": scheduler_type,
            "params": {
                "factor": float(params["factor"]),
                "patience": int(params["patience"]),
                "min_lr": float(params["min_lr"])
            },
            "monitor": scheduler_config.get("monitor", "val/total_loss"),
            "mode": scheduler_config.get("mode", "min")
        }
    elif scheduler_type == "CosineAnnealingLR":
        return {
            "type": scheduler_type,
            "params": {
                "T_max": int(params["T_max"]),
                "eta_min": float(params["eta_min"])
            }
        }
    elif scheduler_type == "CosineAnnealingWarmRestarts":
        return {
            "type": scheduler_type,
            "params": {
                "T_0": int(params["T_0"]),
                "T_mult": int(params["T_mult"]),
                "eta_min": float(params["eta_min"])
            }
        }
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def main(config_path, resume_training=False):

    config = load_yaml_config(config_path)
    log_dir = create_dynamic_log_dir(config)
    train_dataset = torch.load(config['train_dataset'])
    test_dataset = torch.load(config['test_dataset'])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        pin_memory=True,
        num_workers=4,
        collate_fn=collate_fn
    )

    scheduler_config = setup_scheduler(config['scheduler'])
    model = PCNet(
        model_name=MODEL_ID,
        learning_rate=float(config['optimizer']['learning_rate']),
        num_punct_classes=len(punct_label2id),
        num_cap_classes=len(cap_label2id),
        trainable_layers=int(config['layers_to_train']),
        scheduler_config=scheduler_config,
        weight_decay=float(config['optimizer']['weight_decay']),
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=log_dir,
            filename="best-checkpoint",
            save_top_k=1,
            monitor="val/total_loss",
            mode="min",
            save_weights_only=False
        ),
        ModelCheckpoint(
            dirpath=log_dir,
            filename="last-checkpoint",
            save_last=True,
            save_weights_only=False
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]

    if config.get('early_stopping', {}).get('enabled', False):
        callbacks.append(
            EarlyStopping(
                monitor=config['early_stopping'].get('monitor', 'val/total_loss'),
                patience=int(config['early_stopping'].get('patience', 3)),
                mode=config['early_stopping'].get('mode', 'min')
            )
        )

    tb_logger = TensorBoardLogger(
        save_dir=log_dir,
        version=int(config.get("resume_version_tensorboard", None)) if resume_training else None
    )

    ckpt_path = config.get('checkpoint_dir', None) if resume_training else None

    trainer = Trainer(
        max_epochs=config['max_epochs'],
        callbacks=callbacks,
        log_every_n_steps=50,
        default_root_dir=log_dir,
        logger=tb_logger
    )

    trainer.fit(model, train_loader, test_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a model with YAML configuration.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    parser.add_argument(
        "--resume_training",
        action="store_true",
        help="Resume training from a specified checkpoint and logs if set in config."
    )

    args = parser.parse_args()

    main(args.config, resume_training=args.resume_training)

