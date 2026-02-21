"""
Calculate BLEU, ROUGE, and BERTScore from existing generated/reference reports.
"""
import json
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import corpus_bleu
import evaluate

# Load existing reports
with open("generated_reports.json", "r") as f:
    generated_reports = json.load(f)
with open("reference_reports.json", "r") as f:
    reference_reports = json.load(f)

print(f"Loaded {len(generated_reports)} generated reports")
print(f"Loaded {len(reference_reports)} reference reports")

# BLEU scores
print("\n--- BLEU Scores ---")
tokenized_generated = [report.split() for report in generated_reports]
tokenized_references = [[ref.split()] for ref in reference_reports]

bleu1 = corpus_bleu(tokenized_references, tokenized_generated, weights=(1, 0, 0, 0))
bleu2 = corpus_bleu(tokenized_references, tokenized_generated, weights=(0.5, 0.5, 0, 0))
bleu4 = corpus_bleu(tokenized_references, tokenized_generated, weights=(0.25, 0.25, 0.25, 0.25))

print(f"BLEU-1: {bleu1 * 100:.2f}")
print(f"BLEU-2: {bleu2 * 100:.2f}")
print(f"BLEU-4: {bleu4 * 100:.2f}")

# ROUGE scores
print("\n--- ROUGE Scores ---")
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rouge1_scores = []
rouge2_scores = []
rougeL_scores = []

for ref, gen in zip(reference_reports, generated_reports):
    scores = scorer.score(ref, gen)
    rouge1_scores.append(scores['rouge1'].fmeasure)
    rouge2_scores.append(scores['rouge2'].fmeasure)
    rougeL_scores.append(scores['rougeL'].fmeasure)

print(f"ROUGE-1: {sum(rouge1_scores) / len(rouge1_scores) * 100:.2f}")
print(f"ROUGE-2: {sum(rouge2_scores) / len(rouge2_scores) * 100:.2f}")
print(f"ROUGE-L: {sum(rougeL_scores) / len(rougeL_scores) * 100:.2f}")

# BERTScore (optional - takes longer)
print("\n--- BERTScore ---")
print("Calculating BERTScore (this may take a few minutes)...")
bert_score_metric = evaluate.load("bertscore")
bert_results = bert_score_metric.compute(
    predictions=generated_reports,
    references=reference_reports,
    lang="en"
)
print(f"BERTScore Precision: {sum(bert_results['precision']) / len(bert_results['precision']) * 100:.2f}")
print(f"BERTScore Recall: {sum(bert_results['recall']) / len(bert_results['recall']) * 100:.2f}")
print(f"BERTScore F1: {sum(bert_results['f1']) / len(bert_results['f1']) * 100:.2f}")
