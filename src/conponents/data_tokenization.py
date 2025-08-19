from pathlib import Path
import os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

class DataTokenization:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config['model']['tokenizer_name'])
    
    def load_data(self):
        """Load train, validation, and test datasets from parquet files."""
        train_df = pd.read_parquet(os.path.join(self.config["data"]["data_input_dir"], "samsum_train.parquet"))
        validation_df = pd.read_parquet(os.path.join(self.config["data"]["data_input_dir"], "samsum_validation.parquet"))
        test_df = pd.read_parquet(os.path.join(self.config["data"]["data_input_dir"], "samsum_test.parquet"))

        self.train_dataset = Dataset.from_pandas(train_df)
        self.validation_dataset = Dataset.from_pandas(validation_df)
        self.test_dataset = Dataset.from_pandas(test_df)
    
    def tokenize_function(self, batch):
        """Tokenize a batch of dialogues and summaries safely."""
        # Convert all entries to strings to avoid TypeError
        dialogues = [str(d) for d in batch['dialogue']]
        summaries = [str(s) for s in batch['summary']]
        
        # Tokenize inputs
        model_inputs = self.tokenizer(dialogues, max_length=1024, truncation=True)
        
        # Tokenize targets
        labels = self.tokenizer(summaries, max_length=128, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        
        return model_inputs
    
    def data_tokenize(self):
        """Tokenize all datasets."""
        self.tokenized_train = self.train_dataset.map(
            self.tokenize_function, batched=True, remove_columns=["dialogue", "summary"]
        )
        self.tokenized_validation = self.validation_dataset.map(
            self.tokenize_function, batched=True, remove_columns=["dialogue", "summary"]
        )
        self.tokenized_test = self.test_dataset.map(
            self.tokenize_function, batched=True, remove_columns=["dialogue", "summary"]
        )

    def save_to_disk(self):
        """Create directories if not exist and save tokenized datasets."""
        for split in ["train", "validation", "test"]:
            path = Path(self.config["data"]["data_tokenized_dir"][split])
            path.mkdir(parents=True, exist_ok=True)

        self.tokenized_train.save_to_disk(self.config["data"]["data_tokenized_dir"]["train"])
        self.tokenized_validation.save_to_disk(self.config["data"]["data_tokenized_dir"]["validation"])
        self.tokenized_test.save_to_disk(self.config["data"]["data_tokenized_dir"]["test"])
    
    def main(self):
        self.load_data()
        self.data_tokenize()
        self.save_to_disk()