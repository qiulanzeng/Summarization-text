import os
import torch
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate

class ModelEvaluator:
    def __init__(self, config):
        self.config = config

    def batchify(self, lst, batch_size):
        """Yield successive batches from a list."""
        for i in range(0, len(lst), batch_size):
            yield lst[i:i + batch_size]

    def generate_and_decode(self, model, tokenizer, text_batch, device, max_input_length=512, max_output_length=128, num_beams=4):
        """Tokenize input batch, generate summaries, and decode them."""
        inputs = tokenizer(
            text_batch,
            max_length=max_input_length,
            truncation=True,
            padding="longest",
            return_tensors="pt"
        )

        summary_ids = model.generate(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            max_length=max_output_length,
            num_beams=num_beams,
            length_penalty=1.0,
            early_stopping=True
        )

        decoded_summaries = tokenizer.batch_decode(
            summary_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        return decoded_summaries

    def evaluate(self, batch_size=4):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.config['model']['dir_tokenizer'])
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config['model']['dir_model_trained']).to(device)

        # Load test data
        test_df = pd.read_parquet(os.path.join(self.config["data_input_dir"], "samsum_test.parquet"))
        test_dataset = Dataset.from_pandas(test_df)

        # Initialize metrics
        rouge_metric = evaluate.load("rouge")
        bertscore_metric = evaluate.load("bertscore")
        meteor_metric = evaluate.load("meteor")

        all_predictions = []
        all_references = []

        # Batch inference
        for text_batch, ref_batch in tqdm(zip(self.batchify(test_dataset['dialogue'], batch_size),
                                              self.batchify(test_dataset['summary'], batch_size)),
                                          total=len(test_dataset)//batch_size + 1,
                                          desc="Evaluating"):
            decoded_preds = self.generate_and_decode(model, tokenizer, text_batch, device)
            all_predictions.extend(decoded_preds)
            all_references.extend(list(ref_batch))

        # Compute metrics
        rouge_score = rouge_metric.compute(predictions=all_predictions, references=all_references)
        bertscore_score = bertscore_metric.compute(predictions=all_predictions, references=all_references, lang="en")
        meteor_score = meteor_metric.compute(predictions=all_predictions, references=all_references)

        # Extract main ROUGE metrics
        rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        rouge_dict = {rn: rouge_score[rn].mid.fmeasure for rn in rouge_names}

        # Extract BERTScore F1
        bertscore_f1 = sum(bertscore_score["f1"]) / len(bertscore_score["f1"])

        # Combine metrics into single dict
        combined_metrics = {**rouge_dict, "bertscore_f1": bertscore_f1, "meteor": meteor_score["meteor"]}

        # Save metrics to CSV
        df = pd.DataFrame(combined_metrics, index=['pegasus'])
        df.to_csv(self.config.metric_file_name, index=False)

        return combined_metrics