from dotenv import load_dotenv
import mlflow
import json
import pandas as pd
import os
from langchain_openai import ChatOpenAI
from utils import parse_args, set_global_seeds, setup_logger, Config
from workflow.graph import build_graph
from workflow.schema import GraphState

def main():
  load_dotenv()
  args = parse_args()
  set_global_seeds(args.seed)
  logger = setup_logger(Config.LOG_DIR, args.exp_name)

  mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
  mlflow.set_experiment(args.exp_name)

  mlflow.langchain.autolog()

  app = build_graph()

  generator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
  audit_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
  config = {"metadata": {"generator_llm": generator_llm, "audit_llm": audit_llm}}

  result_path = f"{Config.RESULTS_DIR}/{args.exp_name}/{args.run_name}"
  os.makedirs(result_path, exist_ok=True)

  with mlflow.start_run(run_name=args.run_name) as run:
    mlflow.log_params(vars(args))

    initial_state = GraphState(
      personas=[],
      feedback=args.query,
      iterations=0,
      target_count=100,
      seed=args.seed,
      max_iterations=args.max_iterations
    )

    logger.info(f"INITIAL FEEDBACK: {args.query}")

    final_state = app.invoke(initial_state, config=config)
    personas = final_state["personas"]

    personas_dicts = [p.model_dump() for p in personas]
    with open(f"{result_path}/personas.json", "w") as f:
      json.dump(personas_dicts, f, indent=2)
    mlflow.log_artifact(f"{result_path}/personas.json", "personas.json")

    if personas:
        df = pd.DataFrame(personas_dicts)
        mlflow.log_metric("total_generated", len(personas))
        mlflow.log_metric("mean_annual_income", df["annual_income"].mean())
        mlflow.log_metric("disability_rate", df["disability_status"].mean())
        mlflow.log_metric("employed_rate", df["is_employed"].mean())
        mlflow.log_metric("mean_age", df["age"].mean())
        mlflow.log_metric("mean_household_size", df["household_size"].mean())

  logger.info(f"Personas: {len(personas)}")
  logger.info(f"Iterations: {final_state['iterations']}")

if __name__ == "__main__":
  main()