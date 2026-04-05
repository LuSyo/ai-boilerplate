from dotenv import load_dotenv
import os
import mlflow
from langchain_core.messages import HumanMessage
from utils import parse_args, set_global_seeds, setup_logger, Config
from workflow.graph import build_graph
from workflow.schema import GraphState

def main():
  load_dotenv()
  args = parse_args()
  set_global_seeds(args.seed)
  logger = setup_logger(Config.LOG_DIR, args.exp_name)

  result_dir = os.path.join(Config.RESULTS_DIR, args.exp_name)
  os.makedirs(result_dir, exist_ok=True)

  mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
  mlflow.set_experiment(args.exp_name)

  mlflow.langchain.autolog()

  app = build_graph()

  with mlflow.start_run(run_name=args.run_name) as run:
    mlflow.log_params(vars(args))

    initial_state = GraphState(
      messages=[HumanMessage(content=args.query)],
      seed=args.seed
    )

    logger.info(f"QUERY: {args.query}")

    final_state = app.invoke(initial_state)
    answer = final_state["messages"][-1].content

    mlflow.log_text(answer, "answer.txt")


  logger.info(f"ANSWER: {answer}")

if __name__ == "__main__":
  main()