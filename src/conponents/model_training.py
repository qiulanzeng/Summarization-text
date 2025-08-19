from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, AutoTokenizer, EarlyStoppingCallback
from datasets import load_from_disk
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import evaluate
import transformers
print(transformers.__version__)
print(transformers.__file__)


class ModelTrainer:
    def __init__(self, config):
        self.config = config

    def load_tokenized_data(self):
        """Load pre-tokenized datasets from disk."""
        self.tokenized_train = load_from_disk(self.config["data"]["data_tokenized_dir"]["train"])
        self.tokenized_validation = load_from_disk(self.config["data"]["data_tokenized_dir"]["validation"])
        self.tokenized_test = load_from_disk(self.config["data"]["data_tokenized_dir"]["test"])

    def compute_metrics(self, eval_preds):
        """Compute ROUGE scores for seq2seq predictions."""
        rouge = evaluate.load("rouge")
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in labels as padding token id
        labels = [[(l if l != -100 else self.tokenizer.pad_token_id) for l in label] for label in labels]
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        return result

    def main(self):
        self.load_tokenized_data()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model']['tokenizer_name'])
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config['model']['model_name']).to(device)
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=model)

        path = Path(self.config["model"]["ckpt_dir"])
        path.mkdir(parents=True, exist_ok=True)
        
        trainer_args = TrainingArguments(
            output_dir=self.config['model']['ckpt_dir'],
            num_train_epochs=10,
            warmup_steps=500,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            weight_decay=0.01,
            logging_steps=10,
            eval_strategy='steps',
            eval_steps=500,
            save_strategy='steps',
            save_steps=500,
            gradient_accumulation_steps=16,
            load_best_model_at_end=True,
            metric_for_best_model='rougeL',
            greater_is_better=True
        )

        # Early stopping callback
        early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)

        trainer = Trainer(
            model=model,
            args=trainer_args,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            train_dataset=self.tokenized_train,
            eval_dataset=self.tokenized_validation,
            compute_metrics=self.compute_metrics,
            callbacks=[early_stopping_callback],
            predict_with_generate=True
        )

        # Train
        trainer.train()

        # Plot and save metrics
        self.plot_metrics(trainer.state.log_history)

        # Ensure directories exist
        for dir_key in ["model_trained_dir", "tokenizer_dir"]:
            path = Path(self.config["model"][dir_key])
            path.mkdir(parents=True, exist_ok=True)

        # Save best model and tokenizer
        model.save_pretrained(self.config['model']['model_trained_dir'])
        self.tokenizer.save_pretrained(self.config['model']['tokenizer_dir'])

    def plot_metrics(self, logs):
        """Plot training/validation loss and ROUGE metrics, save to model_trained_dir."""
        steps = []
        train_loss = []
        eval_loss = []
        rouge1 = []
        rouge2 = []
        rougeL = []

        for log in logs:
            if "loss" in log:
                steps.append(log["step"])
                train_loss.append(log["loss"])
            if "eval_loss" in log:
                eval_loss.append(log["eval_loss"])
                rouge1.append(log.get("eval_rouge1", 0))
                rouge2.append(log.get("eval_rouge2", 0))
                rougeL.append(log.get("eval_rougeL", 0))

        output_dir = Path(self.config['model']['model_trained_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        # Plot losses
        plt.figure(figsize=(10,5))
        if train_loss:
            plt.plot(steps[:len(train_loss)], train_loss, label="train_loss")
        if eval_loss:
            plt.plot(steps[:len(eval_loss)], eval_loss, label="eval_loss")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.savefig(output_dir / "loss_plot.png")
        plt.close()

        # Plot ROUGE metrics
        if rouge1:
            plt.figure(figsize=(10,5))
            plt.plot(steps[:len(rouge1)], rouge1, label="ROUGE-1")
            plt.plot(steps[:len(rouge2)], rouge2, label="ROUGE-2")
            plt.plot(steps[:len(rougeL)], rougeL, label="ROUGE-L")
            plt.xlabel("Steps")
            plt.ylabel("ROUGE score (%)")
            plt.title("ROUGE Scores During Training")
            plt.legend()
            plt.savefig(output_dir / "rouge_plot.png")
            plt.close()
