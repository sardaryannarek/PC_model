import torch
import sys
sys.path.append('.')
import torch
from torch.utils.data import DataLoader
from core.PC_NET import PCNet
from core.config import punct_label2id, cap_label2id, MODEL_ID


def load_sample_batch(train_dataset_path, batch_size=2):
    dataset = torch.load(train_dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    sample_batch = next(iter(dataloader))
    return sample_batch

def sanity_check(train_dataset_path):
    batch = load_sample_batch(train_dataset_path)

    # Initialize model
    model = PCNet(
        model_name=MODEL_ID,
        learning_rate=1e-4,
        num_punct_classes=len(punct_label2id),
        num_cap_classes=len(cap_label2id),
        trainable_layers=2
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)

    # Forward pass
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    punct_labels = batch["punct_labels"]
    cap_labels = batch["cap_labels"]

    punct_logits, cap_logits = model(input_ids, attention_mask)

    loss = model.compute_loss(punct_logits, cap_logits, punct_labels, cap_labels, attention_mask)

    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    print(f"Punctuation logits shape: {punct_logits.shape}")
    print(f"Capitalization logits shape: {cap_logits.shape}")
    print(f"Loss: {loss.item()}")

train_dataset_path = "scripts/test.pt"

sanity_check(train_dataset_path)
