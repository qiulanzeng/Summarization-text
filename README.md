# End-to-End Text Summarization Project
This is an end-to-end text summarization project. 
1. Data: you can find the datasets from HuggingFace: https://huggingface.co/datasets/knkarthick/samsum. 
2. Model.
    In this project, we trained pegasus-cnn_dailymail for text summarization.
    When using HuggingFace's Trainer with seq2seq models like Pegasus or BART, we should use DataCollatorForSeq2Seq. This collator:
        * padas inputs and labels dynamically per batch (instead of padding everything to a fixed max_length)
        * ensures that the labels are padded with -100 (so the loss ignores padding tokens).
        * is essential for using predict_with_generate=True.
3. Metrics:
    ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
    BLEU (Bilingual Evaluation Understudy)
    BERTScore
4. Deploy method:
    Docker + github action + AWS-EC2

Project Structure.
├───.github
│   └───workflows
├───notebook
├───src 
│   ├───components
│   ├───constants
│   ├───pipeline 
│   ├───logging
│   ├───utils
├───main.py  
├───config.json 

1. config.json: Configuration file that contains paths, project-specific settings and parameters.
2. notebook: Jupyter notebook for experiment.
3. src: Source scripts directory that contains components, 
4. 
1. install all required libraries in requirments.txt
2. write logging file