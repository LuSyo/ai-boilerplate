from dotenv import load_dotenv
import os
import mlflow
import json
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from utils import parse_args, set_global_seeds, setup_logger, Config

mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
mlflow.langchain.autolog(
  log_traces=True, 
  run_tracer_inline=True
)

from workflow.graph import build_graph
from workflow.schema import GraphState

def main():
  # ----- EXPERIMENT SETUP -----
  load_dotenv()
  args = parse_args()
  set_global_seeds(args.seed)
  logger = setup_logger(Config.LOG_DIR, args.exp_name)

  result_dir = os.path.join(Config.RESULTS_DIR, args.exp_name, args.run_name)
  os.makedirs(result_dir, exist_ok=True)

  mlflow.set_experiment(args.exp_name)

  mlflow.langchain.autolog()

  app = build_graph()

  # Set up the RunnableConfig
  explain_risk_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, seed=args.seed)
  eval_llm = ChatOpenAI(model="gpt-4o", temperature=0, seed=args.seed)
  config = RunnableConfig(metadata={"explain_risk_llm": explain_risk_llm,
                                    "eval_llm": eval_llm})

  # ----- START THE RUN -----
  with mlflow.start_run(run_name=args.run_name) as run:
    mlflow.log_params(vars(args))

    initial_state = GraphState(
      seed=args.seed
    )

    final_state = app.invoke(initial_state, config)
    risk_score = final_state["risk_score"]
    shap_dict = final_state["shap_values"]
    explanation = final_state["clinical_explanation"]
    features = final_state["raw_features"]
    evaluation = final_state["evaluation"]
    eval_score = evaluation.score
    eval_feedback = evaluation.feedback
    is_hallucination = evaluation.is_hallucination

    result_payload = {
      "risk_score": risk_score,
      "raw_features": features,
      "shap_values": shap_dict,
      "clinical_explanation": explanation,
      "eval_score": eval_score,
      "eval_feedback": eval_feedback,
      "is_hallucination": is_hallucination
    }

    results_path = os.path.join(result_dir, "results.json")
    with open(results_path, "w") as f:
      json.dump(result_payload, f)

    with open(os.path.join(result_dir, "clinical_briefing.txt"), "w") as f:
      f.write(f"RISK SCORE: {risk_score:.2%}\n\n")
      f.write(f"EXPLANATION:\n{explanation}")

    mlflow.log_metric("eval_score", eval_score)
    mlflow.log_metric("is_hallucination", is_hallucination)
    mlflow.log_metric("risk_score", risk_score)
    mlflow.log_dict(result_payload, "risk_result.json")

  logger.info(f"RISK: {risk_score}")
  logger.info(f"Explanation Logged to: {result_dir}")

if __name__ == "__main__":
  main()