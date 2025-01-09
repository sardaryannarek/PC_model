import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoTokenizer
import pytorch_lightning as pl
from core.config import punct_id2symbol, cap_id2label, inference_text, punct_label2id, cap_label2id
from utils.tokenizer_utils import tokenize_and_align_labels
from utils.reconstruction import reconstruct_sentence_with_word_ids
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts
from optimi import StableAdamW
from sklearn.metrics import classification_report

class PCNet(pl.LightningModule):
    def __init__(self,learning_rate:float, model_name: str, num_punct_classes: int, num_cap_classes: int,
                 trainable_layers: int = 0, scheduler_config=None, weight_decay=1e-2):
        super().__init__()
        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)

        for param in self.bert.parameters():
            param.requires_grad = False

        layers = self.bert._modules['layers']
        total_layers = len(layers)

        if trainable_layers > 0:
            for layer_idx in range(total_layers - trainable_layers, total_layers):
                for param in layers[layer_idx].parameters():
                    param.requires_grad = True

        self.punct_head = nn.Linear(self.config.hidden_size, num_punct_classes)
        self.cap_head = nn.Linear(self.config.hidden_size, num_cap_classes)

        self.punct_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.cap_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.scheduler_config = scheduler_config
        self.initialize_accumulators()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        sequence_output = outputs.last_hidden_state

        punct_logits = self.punct_head(sequence_output)
        cap_logits = self.cap_head(sequence_output)

        return punct_logits, cap_logits

    def compute_loss(self, punct_logits, cap_logits, punct_labels, cap_labels, attention_mask):

        punct_logits = punct_logits.view(-1, self.hparams.num_punct_classes)
        cap_logits = cap_logits.view(-1, self.hparams.num_cap_classes)
        punct_labels = punct_labels.view(-1)
        cap_labels = cap_labels.view(-1)

        active_mask = attention_mask.view(-1) == 1
        active_punct_labels = torch.where(active_mask, punct_labels, torch.tensor(-100).to(self.device))
        active_cap_labels = torch.where(active_mask, cap_labels, torch.tensor(-100).to(self.device))

        punct_loss = self.punct_loss_fn(punct_logits, active_punct_labels)
        cap_loss = self.cap_loss_fn(cap_logits, active_cap_labels)
        total_loss = punct_loss + cap_loss
        return total_loss

    def training_step(self, batch, batch_idx):

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        punct_labels = batch["punct_labels"]
        cap_labels = batch["cap_labels"]

        punct_logits, cap_logits = self(input_ids, attention_mask)

        punct_loss = self.punct_loss_fn(
            punct_logits.view(-1, self.hparams.num_punct_classes),
            punct_labels.view(-1)
        )
        cap_loss = self.cap_loss_fn(
            cap_logits.view(-1, self.hparams.num_cap_classes),
            cap_labels.view(-1)
        )
        total_loss = punct_loss + cap_loss


        self.log("train/punct_loss", punct_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/cap_loss", cap_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        punct_labels = batch["punct_labels"]
        cap_labels = batch["cap_labels"]


        punct_logits, cap_logits = self(input_ids, attention_mask)

        punct_loss = self.punct_loss_fn(
            punct_logits.view(-1, self.hparams.num_punct_classes),
            punct_labels.view(-1)
        )
        cap_loss = self.cap_loss_fn(
            cap_logits.view(-1, self.hparams.num_cap_classes),
            cap_labels.view(-1)
        )
        total_loss = punct_loss + cap_loss
        valid_mask = (attention_mask.view(-1) == 1) & (punct_labels.view(-1) != -100)

        self.punct_preds_all.extend(list(torch.argmax(punct_logits, dim=2).view(-1)[valid_mask].detach().cpu().numpy()))
        self.punct_labels_all.extend(list(punct_labels.view(-1)[valid_mask].detach().cpu().numpy()))

        valid_mask = (attention_mask.view(-1) == 1) & (cap_labels.view(-1) != -100)
        self.cap_preds_all.extend(list(torch.argmax(cap_logits, dim=2).view(-1)[valid_mask].detach().cpu().numpy()))
        self.cap_labels_all.extend(list(cap_labels.view(-1)[valid_mask].detach().cpu().numpy()))

        self.log("val/punct_loss", punct_loss, on_epoch=True, prog_bar=True)
        self.log("val/cap_loss", cap_loss, on_epoch=True, prog_bar=True)
        self.log("val/total_loss", total_loss, on_epoch=True, prog_bar=True)

        return total_loss

    def on_validation_epoch_end(self):

        if self.current_epoch % 5 == 0:

            punct_report = classification_report(
                y_true=self.punct_labels_all, y_pred=self.punct_preds_all, target_names=list(punct_label2id.keys()),
            labels=list(punct_label2id.values()))
            cap_report = classification_report(
                y_true=self.cap_labels_all, y_pred=self.cap_preds_all, target_names=list(cap_label2id.keys()),
            labels=list(cap_label2id.values()))

            self.logger.experiment.add_text(
                tag=f"Report/Cap",
                text_string='\n'.join(cap_report.split('\n')[:-4]),
                global_step=self.current_epoch
            )
            self.logger.experiment.add_text(
                tag=f"Report/Punct",
                text_string='\n'.join(punct_report.split('\n')[:-4]),
                global_step=self.current_epoch
            )
            self.initialize_accumulators()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.to(device)

            predictions = self.predict_on_sentences(inference_text)
            for i, (text, prediction) in enumerate(zip(inference_text, predictions)):
                self.logger.experiment.add_text(
                    tag=f"inference/custom_text_{i}",
                    text_string=f"**Input:** {text}\n\n**Prediction:** {prediction}",
                    global_step=self.current_epoch
                )


    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        optimizer = StableAdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        
        if self.scheduler_config is None:
            return optimizer

        if self.scheduler_config["type"] == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(optimizer, **self.scheduler_config["params"])
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.scheduler_config["monitor"],
                    "interval": "epoch",
                    "frequency": 1
                }
            }
        elif self.scheduler_config["type"] == "CosineAnnealingLR":
            scheduler = CosineAnnealingLR(optimizer, **self.scheduler_config["params"])
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1
                }
            }
        elif self.scheduler_config["type"] == "CosineAnnealingWarmRestarts":
            scheduler = CosineAnnealingWarmRestarts(optimizer, **self.scheduler_config["params"])
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1
                }
            }

    def predict_on_sentences(self, sentences):
        predictions = []
        for sentence in sentences:
            tokens = sentence.split()
            encoding, subword_tokens = tokenize_and_align_labels(
                tokens,
                None,
                None,
                self.tokenizer,
                None,
                None,
                return_labels=False
            )

            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            word_ids = encoding.word_ids(batch_index=0)
            with torch.no_grad():
                punct_logits, cap_logits = self(input_ids, attention_mask)
                punct_preds = torch.argmax(punct_logits, dim=-1).squeeze(0)
                cap_preds = torch.argmax(cap_logits, dim=-1).squeeze(0)

            reconstructed = reconstruct_sentence_with_word_ids(
                subword_tokens,
                word_ids,
                punct_preds,
                cap_preds,
                punct_id2symbol,
                cap_id2label
            )


            predictions.append(reconstructed)
        return predictions

    def initialize_accumulators(self):
        self.punct_preds_all = []
        self.cap_preds_all = []
        self.punct_labels_all = []
        self.cap_labels_all = []