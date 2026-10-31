# Agent State Machine

This diagram is generated directly from the compiled LangGraph `StateGraph`, not hand-drawn. The graph has two regimes: a **fixed pipeline** (`parse_jd` through `generate_report`) that runs once per new/replaced job description, and a conversational **router loop** (`agent_router`) that handles every follow-up chat turn by calling tools (compare, explain, re-rank, interview questions, final recommendation) without re-running the pipeline. Both paths converge at `human_feedback`, which simply marks the end of a turn — each chat message is one `graph.invoke()` against a SQLite-checkpointed thread, which is how conversation state persists across turns.

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	parse_jd(parse_jd)
	extract_requirements(extract_requirements)
	search_resumes(search_resumes)
	rank_candidates(rank_candidates)
	generate_report(generate_report)
	agent_router(agent_router)
	human_feedback(human_feedback)
	__end__([<p>__end__</p>]):::last
	__start__ -.-> agent_router;
	__start__ -.-> parse_jd;
	agent_router --> human_feedback;
	extract_requirements --> search_resumes;
	generate_report --> human_feedback;
	parse_jd --> extract_requirements;
	rank_candidates --> generate_report;
	search_resumes --> rank_candidates;
	human_feedback --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc

```
