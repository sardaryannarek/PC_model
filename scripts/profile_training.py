import sys
sys.path.append('.')
import yaml
import os
import torch
from torch.utils.data import DataLoader
from core.PC_NET import PCNet
from core.config import punct_label2id, cap_label2id, MODEL_ID

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch._dynamo.config.cache_size_limit = 256
torch.set_float32_matmul_precision('high')

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


def train_one_epoch(model, train_loader, optimizer, device):
    model.train()
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        punct_labels = batch["punct_labels"].to(device)
        cap_labels = batch["cap_labels"].to(device)

        optimizer.zero_grad()
        punct_logits, cap_logits = model(input_ids, attention_mask)
        loss = model.compute_loss(punct_logits, cap_logits, punct_labels, cap_labels, attention_mask)
        loss.backward()
        optimizer.step()


def main(config_path):
    config = load_yaml_config(config_path)
    log_dir = create_dynamic_log_dir(config)

    train_dataset = torch.load(config['train_dataset'])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        pin_memory=True,
        num_workers=8
    )

    model = PCNet(
        model_name=MODEL_ID,
        learning_rate=config['optimizer']['learning_rate'],
        num_punct_classes=len(punct_label2id),
        num_cap_classes=len(cap_label2id),
        trainable_layers=config['layers_to_train']
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['optimizer']['learning_rate']))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ],
        schedule=torch.profiler.schedule(wait=0, warmup=1, active=5, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
        record_shapes=True,
        with_stack=True
    ) as prof:
        for epoch in range(1):
            train_one_epoch(model, train_loader, optimizer, device)

    if prof:
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        print(f"Profiling logs saved to {log_dir}. Use TensorBoard to visualize.")
    else:
        print("Profiler did not collect any data. Check schedule and DataLoader.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Profile a model with YAML configuration.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    args = parser.parse_args()

    main(args.config)
