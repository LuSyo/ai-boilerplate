from dotenv import load_dotenv
import os
import mlflow
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
  generate_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, seed=args.seed)
  config = RunnableConfig(metadata={"generate_llm": generate_llm})

  # ----- START THE RUN -----
  with mlflow.start_run(run_name=args.run_name) as run:
    mlflow.log_params(vars(args))

    initial_state = GraphState(
      messages=[HumanMessage(content=args.query)],
      seed=args.seed
    )

    logger.info(f"QUERY: {args.query}")

    final_state = app.invoke(initial_state, config)
    answer = final_state["messages"][-1].content

    mlflow.log_text(answer, "answer.txt")


  logger.info(f"ANSWER: {answer}")

if __name__ == "__main__":
  main()