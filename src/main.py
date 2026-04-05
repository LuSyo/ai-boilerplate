from dotenv import load_dotenv
import os
import json
import mlflow
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from utils import parse_args, set_global_seeds, setup_logger, Config

mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
mlflow.langchain.autolog(
  log_traces=True, 
  run_tracer_inline=True
)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from workflow.graph import build_graph
from workflow.schema import GraphState

def main():
  # ---- Experiment setup
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

  # Set up the RunnableConfig
  triage_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
  response_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
  embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

  config = {"metadata": {"triage_llm": triage_llm, 
                         "embeddings": embeddings, 
                         "response_llm": response_llm}}
  
  # ---- Start the run
  with mlflow.start_run(run_name=args.run_name) as run:
    mlflow.log_params(vars(args))

    results =[]

    # Process one email at a time
    for filename in incoming_files:
      file_path = os.path.join(incoming_dir, filename)
      with open(file_path, "r") as f:
        email_content = f.read()

      initial_state = GraphState(
        raw_message=email_content,
        seed=args.seed
      )

      logger.info(f"EMAIL: {email_content[:30]}")

      # Run the workflow
      final_state = app.invoke(initial_state, config=config) 
      draft_response = final_state["draft"]

      # Log the results
      results.append({
        "filename": filename,
        "category_pred": final_state["category"]
      })

      # Save the draft response
      if draft_response:
        file_path = os.path.join(drafts_dir, f"{args.run_name}_{filename}_response.txt")
        with open(file_path, "w") as f:
          f.write(draft_response)

      mlflow.log_artifact(file_path, f"{args.run_name}_{filename}_response.txt")

    #---- Performance metrics
    eval_gt_path = os.path.join(Config.EVAL_DIR, "email_eval_gt.json")
    with open(eval_gt_path, "r") as f:
      ground_truth = json.load(f)

    y_true = [ground_truth[r["filename"]] for r in results]
    y_pred = [r["category_pred"] for r in results]
    labels = ["query", "urgent", "spam", "unknown"]

    metrics, confusion_matrix_fig, class_report = eval_metrics(y_true, y_pred, labels)
    mlflow.log_metrics(metrics)
    mlflow.log_figure(confusion_matrix_fig, "confusion_matrix.png")

    with open(f"{result_dir}/metrics.json", "w") as f:
      json.dump(metrics, f)
    with open(f"{result_dir}/class_report.json", "w") as f:
      json.dump(class_report, f)

    confusion_matrix_fig.savefig(f"{result_dir}/confusion_matrix.png")



def eval_metrics(y_true, y_pred, labels):
  # Classification report
  report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=np.nan)
  print("Classification Report:")
  print(report)
  
  # Generate Confusion Matrix
  cm = confusion_matrix(y_true, y_pred, labels=labels)
  fig, ax = plt.subplots(figsize=(8, 6))
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
  disp.plot(cmap=plt.cm.Blues, ax=ax)
  ax.set_title("Email Triage Confusion Matrix")
  
  # Calculate accuracy
  correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
  accuracy = correct / len(y_true) if y_true else 0
  
  return {"accuracy": accuracy}, fig, report

if __name__ == "__main__":
  main()