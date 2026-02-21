import json
import re
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer

# -----------------------------------------------------------------------------
# 1. Load the generated and reference reports from JSON files
# -----------------------------------------------------------------------------
with open("generated_reports.json", "r", encoding="utf-8") as f:
    generated_reports = json.load(f)

with open("reference_reports.json", "r", encoding="utf-8") as f:
    reference_reports = json.load(f)

# For BLEU scores, tokenize by splitting on whitespace.
tokenized_generated = [report.split() for report in generated_reports]
# Note: corpus_bleu expects a list of reference lists for each sample.
tokenized_references = [[ref.split()] for ref in reference_reports]

# -----------------------------------------------------------------------------
# 3. Compute BLEU Scores
# -----------------------------------------------------------------------------
# BLEU-1: using only unigrams
bleu1 = corpus_bleu(tokenized_references, tokenized_generated, weights=(1, 0, 0, 0))
print(f"BLEU-1: {bleu1 * 100:.2f}")

# BLEU-4: using up to 4-grams
bleu4 = corpus_bleu(tokenized_references, tokenized_generated, weights=(0.25, 0.25, 0.25, 0.25))
print(f"BLEU-4: {bleu4 * 100:.2f}")

# -----------------------------------------------------------------------------
# 4. Compute ROUGE Scores
# -----------------------------------------------------------------------------
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
