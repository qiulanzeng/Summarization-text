#load the Samsum dataset from Hugging Face Hub
from datasets import load_dataset
from pathlib import Path
import os

class DataDownload:
    def __init__(self, config):
        self.config = config

    def main(self):
        # Load the dataset
        dataset = load_dataset("knkarthick/samsum")

        # Access splits
        train_data = dataset["train"]
        test_data = dataset["test"]
        validation_data = dataset["validation"]

        #save locally as Parquet
        Path(self.config["data"]["data_input_dir"]).mkdir(parents=True, exist_ok=True)
        train_data.to_parquet(os.path.join(self.config["data"]["data_input_dir"],"samsum_train.parquet"))
        test_data.to_parquet(os.path.join(self.config["data"]["data_input_dir"],"samsum_test.parquet"))
        validation_data.to_parquet(os.path.join(self.config["data"]["data_input_dir"],"samsum_validation.parquet"))



    def data_download_stream(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        #if the training dataset is large, you can stream it in chunks, save to Parquet, and tokenize without loadng everything into memory at once.
        dataset = load_dataset("samsum", split="train", streaming=True)
        batch_size = 1000
        buffer = []

        for i, example in enumerate(dataset):
            buffer.append(example)
            
            if (i + 1) % batch_size == 0:
                table = pa.Table.from_pylist(buffer)
                pq.write_to_dataset(table, root_path="samsum_train_parquet", partition_cols=None)
                buffer = []

        # Write remaining rows
        if buffer:
            table = pa.Table.from_pylist(buffer)
            pq.write_to_dataset(table, root_path="samsum_train_parquet")