from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from utils import parse_args, set_global_seeds, setup_logger, Config
from workflow.graph import build_graph
from workflow.schema import GraphState

def main():
  load_dotenv()
  args = parse_args()
  set_global_seeds(args.seed)
  logger = setup_logger(Config.LOG_DIR, args.exp_name)

  app = build_graph()

  initial_state = GraphState(
    messages=[HumanMessage(content=args.query)],
    seed=args.seed
  )

  logger.info(f"QUERY: {args.query}")

  final_state = app.invoke(initial_state)
  answer = final_state["messages"][-1].content

  logger.info(f"ANSWER: {answer}")

if __name__ == "__main__":
  main()