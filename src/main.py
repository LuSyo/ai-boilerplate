from dotenv import load_dotenv
import os
import json
import mlflow
import numpy as np
from utils import parse_args, set_global_seeds, setup_logger, Config

mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
mlflow.langchain.autolog(
  log_traces=True, 
  run_tracer_inline=True
)

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from workflow.graph import build_graph
from workflow.schema import GraphState

def main():
  load_dotenv()
  args = parse_args()
  set_global_seeds(args.seed)
  logger = setup_logger(Config.LOG_DIR, args.exp_name)

  result_dir = os.path.join(Config.RESULTS_DIR, args.exp_name, args.run_name)
  os.makedirs(result_dir, exist_ok=True)
  drafts_dir = os.path.join(Config.DATA_DIR, "drafts", args.exp_name, args.run_name)
  os.makedirs(drafts_dir, exist_ok=True)

  incoming_dir = os.path.join(Config.DATA_DIR, "incoming")
  incoming_files = [f for f in os.listdir(incoming_dir) if f.endswith('.txt')]
  
  mlflow.set_experiment(args.exp_name)  

  app = build_graph()

  triage_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
  response_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
  embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

  config = {"metadata": {"triage_llm": triage_llm, 
                         "embeddings": embeddings, 
                         "response_llm": response_llm}}

  with mlflow.start_run(run_name=args.run_name) as run:
    mlflow.log_params(vars(args))

    results =[]
    for filename in incoming_files:

      file_path = os.path.join(incoming_dir, filename)
      with open(file_path, "r") as f:
        email_content = f.read()

      initial_state = GraphState(
        raw_message=email_content,
        seed=args.seed
      )

      logger.info(f"EMAIL: {email_content[:30]}")

      final_state = app.invoke(initial_state, config=config) 
      draft_response = final_state["draft"]

      results.append({
        "filename": filename,
        "category_pred": final_state["category"]
      })

      if draft_response:
        file_path = os.path.join(drafts_dir, f"{args.run_name}_{filename}_response.txt")
        with open(file_path, "w") as f:
          f.write(draft_response)

      mlflow.log_artifact(file_path, f"{args.run_name}_{filename}_response.txt")

    eval_gt_path = os.path.join(Config.EVAL_DIR, "email_eval_gt.json")
    metrics = eval_metrics(results, eval_gt_path)
    mlflow.log_metrics(metrics)

    with open(f"{result_dir}/metrics.json", "w") as f:
      json.dump(metrics, f)


def eval_metrics(results, gt_path):
  with open(gt_path, "r") as f:
    ground_truth = json.load(f)

  # with regards to "query" class
  fn_q, fp_q, tn_q, tp_q = [0]*4
  # with regards to "urgent" class
  fn_u, fp_u, tn_u, tp_u = [0]*4

  correct = 0

  for r in results:
    filename = r["filename"] 
    prediction = r["category_pred"]
    if prediction == ground_truth[filename]:
      correct += 1

    if ground_truth[filename] == "query":
      if prediction  == "query":
        tp_q += 1
      else:
        fn_q +=1
    else:
      if prediction == "query":
        fp_q += 1
      else:
        tn_q +=1

    if ground_truth[filename] == "urgent":
      if prediction == "urgent":
        tp_u += 1
      else:
        fn_u +=1
    else:
      if prediction == "urgent":
        fp_u += 1
      else:
        tn_u +=1
  
  precision_query = tp_q / (tp_q + fp_q) if tp_q + fp_q > 0 else np.nan
  recall_query = tp_q / (tp_q + fn_q)
  precision_urgent = tp_u / (tp_u + fp_u) if tp_u + fp_u > 0 else np.nan
  recall_urgent = tp_u / (tp_u + fn_u)

  accuracy = correct / len(results)

  return {
    "accuracy": accuracy, "precision_query": precision_query, "recall_query": recall_query, "precision_urgent": precision_urgent, "recall_urgent": recall_urgent}


if __name__ == "__main__":
  main()