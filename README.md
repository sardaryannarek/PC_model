

```markdown
# Model Training and Inference Guide

This guide explains how to use the provided scripts for training and inference of the punctuation and capitalization model.

---

## Inference

### How to Run the Inference Script

To run the inference script, use the following command:

```bash
python inference_script.py --checkpoint_dir <path_to_model_checkpoint> --input_txt_file <path_to_input_text> --window_size <chunk_size> --stride <overlap_size> --output_txt_file <path_to_output_text>
```

### Arguments
- `--checkpoint_dir`: Path to the model checkpoint (e.g., `logs/best-checkpoint.ckpt`).
- `--input_txt_file`: Path to the input text file containing the text to process.
- `--window_size` (optional): Number of words per chunk. Default is `49`.
- `--stride` (optional): Overlap in terms of words. Default is `25`.
- `--output_txt_file` (optional): Path to save the processed text. If not provided, the output will be printed to the console.

### Example
```bash
python inference_script.py --checkpoint_dir models/best-checkpoint.ckpt --input_txt_file data/sample.txt --output_txt_file results/output.txt
```

---

## Training

### How to Run the Training Script

To train the model, use the following command:

```bash
python training_script.py --config <path_to_config_file> --resume_training
```

### Arguments
- `--config`: Path to the YAML configuration file (e.g., `config/train_config.yml`).
- `--resume_training` (optional): If provided, resumes training from the specified checkpoint in the configuration file.

### Example
```bash
python training_script.py --config config/train_config.yml --resume_training
```

---

## Editing the Configuration File

The training process can be customized by editing the provided YAML configuration file. Below is an example configuration file:

```yaml
# General Training Settings
train_dataset: "data/pt_datasets/train.pt"
test_dataset: "data/pt_datasets/test.pt"
batch_size: 512
max_epochs: 300
checkpoint_dir: 'logs_all/logs/PCNet_lr1e-4_layers0/last-v1.ckpt'
resume_version_tensorboard: 0

# Early Stopping Settings
early_stopping:
  enabled: true
  patience: 10
  monitor: "val/total_loss"
  mode: "min"

# Optimizer Settings
optimizer:
  type: "AdamW"
  learning_rate: 5e-5
  weight_decay: 3e-2

# Scheduler Settings
scheduler:
  enabled: false
  type: "CosineAnnealingLR"
  params:
    T_max: 25

# Model Settings
layers_to_train: 0

# Logging and Saving
logging:
  dynamic_log_dir: true
  log_dir_base: "test_logs"
  tensorboard_enabled: true
  checkpoint_save_interval: 1
```

### Key Parameters to Edit:
1. **Training Dataset**: Update `train_dataset` and `test_dataset` paths to point to your datasets.
2. **Batch Size**: Adjust `batch_size` as per your system’s capacity.
3. **Learning Rate**: Modify `learning_rate` under `optimizer` for experiments.
4. **Scheduler**: Enable and configure the learning rate scheduler under `scheduler`.
5. **Checkpoint Directory**: Set `checkpoint_dir` to resume training from a specific checkpoint or leave it blank for fresh training.

---

## Notes
- Ensure the required libraries are installed before running the scripts.
- Use `scripts/always_predictable.yml` to set rules for capitalization and punctuation adjustments.

```
