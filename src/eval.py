import re
import os
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from utils import parse_args, set_global_seeds, setup_logger, Config
from typing import List, Tuple
from workflow.graph import build_graph
from workflow.schema import GraphState

def eval():
  load_dotenv()
  args = parse_args()
  set_global_seeds(args.seed)
  logger = setup_logger(Config.LOG_DIR, args.exp_name)

  app = build_graph()
  results = []
  redacted_samples = []

  results_path = os.path.join(Config.RESULTS_DIR, args.exp_name)
  os.makedirs(results_path, exist_ok=True)

  dataset_path = os.path.join(Config.EVAL_DIR, args.eval_dataset)
  with open(dataset_path, 'r') as f:
        eval_set = json.load(f)

  logger.info(f"--- Starting Eval on {len(eval_set)} cases ---")
  
  for entry in eval_set:
    text = entry["text"]
    ground_truth_text = entry["ground_truth"]

    print(f"ENTRY ID: {entry['id']}")

    initial_state = GraphState(
      messages=[],
      text=text,
      original_text=text,
      pii_registry=[],
      seed=args.seed
    )

    final_state = app.invoke(initial_state)

    result = calculate_redaction_metrics(
      output_text=final_state["text"],
      ground_truth_text=ground_truth_text
    )

    results.append(result)

    redacted_samples.append({
      "id": entry["id"],
      "text": text,
      "ground_truth": ground_truth_text,
      "redacted": final_state["text"],
    } | result)
  
  results_df = pd.DataFrame(results)

  avg_recall = results_df["recall"].mean()
  avg_precision = results_df["precision"].mean()
  total_missed = results_df["missed_count"].sum()
  total_extra = results_df["extra_count"].sum()

  logger.info(f"Average Recall: {avg_recall}")
  logger.info(f"Average Precision: {avg_precision}")
  logger.info(f"Total Missed (False Negatives): {total_missed}")
  logger.info(f"Total Extra (False Positives): {total_extra}")

  report_path = os.path.join(results_path, f"redaction_report.json")
  with open(report_path, 'w') as f:
      json.dump(redacted_samples, f, indent=2)


def get_label_spans(text: str) -> List[Tuple[int, int, str]]:
  """Finds the start/end offsets and content of all [LABEL] tags."""
  pattern = r"\[([A-Z0-9_]+)\]"
  return [(m.start(), m.end(), m.group(0)) for m in re.finditer(pattern, text)]

def calculate_redaction_metrics(output_text: str, ground_truth_text: str, 
                              affordance: int = 1):
  """
  Compares redaction spans with a positional affordance.
  """
  output_spans = get_label_spans(output_text)
  truth_spans = get_label_spans(ground_truth_text)

  print(len(truth_spans))
  
  matches = 0
  misses = 0
  extras = 0

  # Check for matches/misses
  for t_start, t_end, _ in truth_spans:
      # Look for a label in the output that is within 'affordance' characters
      found_start = any(abs(o_start - t_start) <= affordance for o_start, _, _ in output_spans)
      found_end = any(abs(o_end - t_end) <= affordance for _, o_end, _ in output_spans)
      found = found_start and found_end
      if found:
          matches += 1
      else:
          misses += 1

  # Check for over-redaction (labels in output not in truth)
  for o_start, o_end, _ in output_spans:
      found_start = any(abs(t_start - o_start) <= affordance for t_start, _, _ in truth_spans)
      found_end = any(abs(t_end - o_end) <= affordance for _, t_end, _ in truth_spans)
      found = found_start and found_end
      if not found:
          extras += 1

  recall = matches / len(truth_spans) if truth_spans else np.nan
  precision = matches / (matches + extras) if (matches + extras) > 0 else np.nan
  
  return {
      "recall": recall,       # Safety: Did we catch what we should?
      "precision": precision, # Utility: Did we redact only what was necessary?
      "missed_count": misses,
      "extra_count": extras
  }

if __name__ == "__main__":
  eval()
  