import os
import sys
import json
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
from tqdm import tqdm
import torch
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
from transformers import TextStreamer
from unsloth import FastVisionModel
from dataset import get_data_loaders
import re
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import corpus_bleu
import evaluate

# Append current directory to sys.path (if needed)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 1. Load the dataset (including the test loader)
cleaned_dataset_dir = os.environ.get("DATASET_DIR", "data/cleaned_dataset")
output_excel_file = os.environ.get("METADATA_FILE", "data/cleaned_dataset_metadata.xlsx")
_, _, test_loader = get_data_loaders(output_excel_file, cleaned_dataset_dir, drop_last=True)

# 2. Load the trained model and tokenizer from the saved checkpoint.
#    Note: We assume the checkpoint already includes LoRA adapters, so we don't reapply them.
model, tokenizer = FastVisionModel.from_pretrained(
    "outputs/checkpoint-2516",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth"
)

# Set the model to inference mode.
FastVisionModel.for_inference(model)

# Define your instruction and conversation template (same for all samples in this example)
instruction = ("You are an expert radiographer. Describe accurately what you see in this chest x-ray image. "
               "Write it as you write the findings section of the report.")
conversation_template = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": instruction}
    ]}
]

# We'll generate reports for the first 100 samples from the test loader.
num_samples = len(test_loader)
sample_count = 0

print("\nGenerating reports for {} samples:\n".format(num_samples))
generated_reports = []
reference_reports = []

pbar = tqdm(total=num_samples, desc="Processing Samples")
for batch in test_loader:
    # If your batch contains multiple samples, you can loop over them.
    batch_size = len(batch["PA"])
    for i in range(batch_size):
        """
        if sample_count >= num_samples:
            break
        """
        # Get the image and actual finding
        image = batch["PA"][i]
        actual_finding = batch["Findings"][i]

        # Convert image tensor to a PIL image
        pil_image = to_pil_image(image)

        # Create the chat prompt using the tokenizer's chat template method
        input_text = tokenizer.apply_chat_template(conversation_template, add_generation_prompt=True)

        # Tokenize the inputs; ensure they are sent to the same device as the model.
        inputs = tokenizer(
            pil_image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda")

        #print(f"Sample {sample_count + 1}:")
        text_streamer = TextStreamer(tokenizer, skip_prompt=True)
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            use_cache=True,
            do_sample=True,
            temperature=1.5,
            top_p=0.8
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        parts = generated_text.split("FINDINGS:", 1)
        generated_findings = parts[1].strip() if len(parts) > 1 else ""
        generated_findings = re.sub(r'\s+', ' ', generated_findings).strip()
        generated_reports.append(generated_findings)
        actual_finding = re.sub(r'\s+', ' ', actual_finding).strip()
        parts2 = actual_finding.split("FINDINGS:", 1)
        actual_finding = parts2[1].strip() if len(parts) > 1 else ""
        reference_reports.append(actual_finding)

        sample_count += 1
        pbar.update(1)

# Optionally, save the outputs to JSON files.
with open("generated_reports.json", "w", encoding="utf-8") as f:
    json.dump(generated_reports, f, ensure_ascii=False, indent=2)
with open("reference_reports.json", "w", encoding="utf-8") as f:
    json.dump(reference_reports, f, ensure_ascii=False, indent=2)

print(f"\nGenerated reports for {len(generated_reports)} samples saved to 'generated_reports.json' and 'reference_reports.json'.")

# -----------------------------------------------------------------------------
# 4. Compute Similarity Metrics Using the Evaluate Library
# -----------------------------------------------------------------------------
print("\n--- Similarity Metrics ---")
# Compute BLEU scores
tokenized_generated = [report.split() for report in generated_reports]
tokenized_references = [[ref.split()] for ref in reference_reports]
bleu1 = corpus_bleu(tokenized_references, tokenized_generated, weights=(1, 0, 0, 0))
print(f"BLEU-1: {bleu1 * 100:.2f}")

bleu4 = corpus_bleu(tokenized_references, tokenized_generated, weights=(0.25, 0.25, 0.25, 0.25))
print(f"BLEU-4: {bleu4 * 100:.2f}")

# Compute ROUGE scores
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
rouge1_scores = []
rougeL_scores = []

for ref, cand in zip(reference_reports, generated_reports):
    scores = scorer.score(ref, cand)
    rouge1_scores.append(scores['rouge1'].fmeasure)
    rougeL_scores.append(scores['rougeL'].fmeasure)

avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)

print(f"Average ROUGE-1 F-measure: {avg_rouge1:.2f}")
print(f"Average ROUGE-L F-measure: {avg_rougeL:.2f}")

# Compute BERTScore
bert_score_metric = evaluate.load("bertscore")
bert_score = bert_score_metric.compute(predictions=generated_reports,
                                         references=reference_reports,
                                         lang="en")
print("\nBERTScore:")
print(bert_score)
