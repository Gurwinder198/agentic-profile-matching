"""Terminal chat interface for the recruiting agent."""

import os
import sys
import uuid

import matching_agent


def main() -> None:
    checkpoint_path = os.environ.get("CHECKPOINT_PATH", "data/checkpoints.sqlite")
    graph = matching_agent.build_graph(checkpoint_path)
    thread_id = str(uuid.uuid4())

    for raw_line in sys.stdin:
        line = raw_line.rstrip("\n").strip()
        if not line:
            continue
        if line.lower() in ("quit", "exit"):
            break

        if line.startswith("JD:"):
            job_description = line[len("JD:"):].strip()
            result = matching_agent.run_turn(graph, thread_id, job_description=job_description)
        else:
            result = matching_agent.run_turn(graph, thread_id, user_message=line)

        print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
