"""Streamlit web UI for the Tool Router Simulator."""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import streamlit as st

from .config import (
    ALL_MODELS,
    VERCEL_MODELS,
    GROQ_MODELS,
    BEDROCK_MODELS,
    Provider,
    ModelConfig,
)
from .composio_client import create_client, ComposioClient
from .simulator.runner import SimulationResult
from .simulator.vercel_runner import create_vercel_runner
from .simulator.groq_runner import create_groq_runner
from .simulator.bedrock_runner import create_bedrock_runner
from .evaluator.scorer import CompositeScorer, BenchmarkScore
from .benchmark.tasks import get_builtin_suite, get_all_builtin_tasks


def get_runner_for_model(model_name: str, client: ComposioClient):
    """Create the appropriate runner for a model."""
    if model_name in VERCEL_MODELS:
        return create_vercel_runner(client, model_name)
    elif model_name in GROQ_MODELS:
        return create_groq_runner(client, model_name)
    elif model_name in BEDROCK_MODELS:
        return create_bedrock_runner(client, model_name)
    else:
        config = ALL_MODELS.get(model_name)
        if config:
            if config.provider == Provider.VERCEL:
                return create_vercel_runner(client, model_name)
            elif config.provider == Provider.GROQ:
                return create_groq_runner(client, model_name)
            elif config.provider == Provider.BEDROCK:
                return create_bedrock_runner(client, model_name)
        raise ValueError(f"Unknown model: {model_name}")


def run_simulation(task: str, model_name: str, client: ComposioClient):
    """Run a simulation for a single model."""
    try:
        runner = get_runner_for_model(model_name, client)
        result = runner.simulate(task)
        scorer = CompositeScorer()
        score = scorer.score(result)
        return model_name, result, score, None
    except Exception as e:
        return model_name, None, None, str(e)


def grade_color(grade: str) -> str:
    """Get color for a grade."""
    if grade.startswith("A"):
        return "green"
    elif grade.startswith("B"):
        return "blue"
    elif grade.startswith("C"):
        return "orange"
    else:
        return "red"


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Tool Router Simulator",
        page_icon="🔧",
        layout="wide",
    )

    st.title("🔧 Composio Tool Router Simulator")
    st.markdown("Benchmark how different LLM models perform with Composio's Tool Router")

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Dry run mode
        dry_run = st.checkbox("🧪 Dry Run Mode", value=True, help="Test without making API calls")

        st.subheader("Select Models")

        # Group models by provider
        st.markdown("**Vercel AI Gateway**")
        vercel_selected = []
        for name, config in VERCEL_MODELS.items():
            if st.checkbox(f"{config.name}", key=f"vercel_{name}", value=True):
                vercel_selected.append(name)

        st.markdown("**Groq (Fast Inference)**")
        groq_selected = []
        for name, config in GROQ_MODELS.items():
            if st.checkbox(f"{config.name}", key=f"groq_{name}", value=name in ["llama-3.3-70b", "llama-3.1-8b"]):
                groq_selected.append(name)

        st.markdown("**AWS Bedrock**")
        bedrock_selected = []
        for name, config in BEDROCK_MODELS.items():
            if st.checkbox(f"{config.name}", key=f"bedrock_{name}", value=False):
                bedrock_selected.append(name)

        selected_models = vercel_selected + groq_selected + bedrock_selected

        st.divider()
        st.markdown(f"**Selected:** {len(selected_models)} models")

    # Main content
    tab1, tab2, tab3 = st.tabs(["🚀 Run Task", "📊 Benchmark Suite", "📋 Results History"])

    with tab1:
        st.subheader("Run a Single Task")

        # Task input
        task = st.text_area(
            "Enter your task:",
            placeholder="e.g., List my 5 most recent unread emails",
            height=100,
        )

        # Example tasks
        with st.expander("💡 Example Tasks"):
            examples = [
                "List my 5 most recent unread emails",
                "Send a Slack message to #general saying 'Hello team!'",
                "Create a new GitHub issue titled 'Bug fix needed'",
                "Find all messages in #engineering from today",
                "Add a row to my Google Sheet with today's date",
            ]
            for ex in examples:
                if st.button(ex, key=f"ex_{ex[:20]}"):
                    st.session_state.task = ex
                    st.rerun()

        # Run button
        col1, col2 = st.columns([1, 4])
        with col1:
            run_button = st.button("▶️ Run Benchmark", type="primary", disabled=not task or not selected_models)

        if run_button and task and selected_models:
            st.divider()
            st.subheader("Results")

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Create client
            client = create_client(dry_run=dry_run)

            # Run simulations
            results = {}
            total = len(selected_models)

            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {
                    executor.submit(run_simulation, task, model, client): model
                    for model in selected_models
                }

                completed = 0
                for future in as_completed(futures):
                    model_name, result, score, error = future.result()
                    completed += 1
                    progress_bar.progress(completed / total)
                    status_text.text(f"Completed: {model_name} ({completed}/{total})")

                    if error:
                        results[model_name] = {"error": error}
                    else:
                        results[model_name] = {"result": result, "score": score}

            progress_bar.empty()
            status_text.empty()

            # Display results table
            st.markdown("### 📊 Benchmark Results")

            # Sort by score
            sorted_results = sorted(
                [(m, r) for m, r in results.items() if "score" in r],
                key=lambda x: x[1]["score"].overall_score,
                reverse=True,
            )

            # Create results table
            if sorted_results:
                cols = st.columns([3, 2, 2, 2, 2, 1])
                cols[0].markdown("**Model**")
                cols[1].markdown("**Tool Selection**")
                cols[2].markdown("**Execution**")
                cols[3].markdown("**Speed**")
                cols[4].markdown("**Cost**")
                cols[5].markdown("**Grade**")

                st.divider()

                for model, data in sorted_results:
                    score = data["score"]
                    metrics = score.details.get("metrics", {})

                    cols = st.columns([3, 2, 2, 2, 2, 1])
                    cols[0].write(model)
                    cols[1].write(f"{score.tool_selection.score * 100:.0f}%")
                    cols[2].write(f"{score.execution.score * 100:.0f}%")

                    time_ms = metrics.get("total_time_ms", 0)
                    cols[3].write(f"{time_ms/1000:.1f}s" if time_ms else "N/A")

                    cost = metrics.get("estimated_cost_usd", 0)
                    cols[4].write(f"${cost:.4f}" if cost else "N/A")

                    grade = score.grade
                    cols[5].markdown(f"**:{grade_color(grade)}[{grade}]**")

                # Best in category
                st.divider()
                col1, col2, col3, col4 = st.columns(4)

                if sorted_results:
                    col1.metric("🏆 Best Overall", sorted_results[0][0])

                    fastest = min(
                        [(m, r["score"].details.get("metrics", {}).get("total_time_ms", float("inf")))
                         for m, r in sorted_results],
                        key=lambda x: x[1]
                    )
                    col2.metric("⚡ Fastest", fastest[0])

                    cheapest = min(
                        [(m, r["score"].details.get("metrics", {}).get("estimated_cost_usd", float("inf")))
                         for m, r in sorted_results],
                        key=lambda x: x[1]
                    )
                    col3.metric("💰 Cheapest", cheapest[0])

                    best_tools = max(sorted_results, key=lambda x: x[1]["score"].tool_selection.score)
                    col4.metric("🎯 Best Tool Selection", best_tools[0])

            # Show errors
            errors = [(m, r["error"]) for m, r in results.items() if "error" in r]
            if errors:
                with st.expander(f"⚠️ Errors ({len(errors)})", expanded=False):
                    for model, error in errors:
                        st.error(f"**{model}:** {error}")

            # Detailed results
            with st.expander("📋 Detailed Results", expanded=False):
                for model, data in sorted_results:
                    st.markdown(f"#### {model}")
                    score = data["score"]

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Tool Selection:**")
                        st.write(f"- Found relevant tools: {'✅' if score.tool_selection.found_relevant_tools else '❌'}")
                        st.write(f"- Selected correct tool: {'✅' if score.tool_selection.selected_correct_tool else '❌'}")
                        queries = score.details.get("search_queries", [])
                        if queries:
                            st.write(f"- Search query: `{queries[0]}`")

                    with col2:
                        st.markdown("**Execution:**")
                        st.write(f"- Successful: {'✅' if score.execution.successful_execution else '❌'}")
                        st.write(f"- Correct params: {'✅' if score.execution.correct_parameters else '❌'}")
                        selected = score.details.get("tools_selected", [])
                        if selected:
                            st.write(f"- Tools: {', '.join(selected)}")

                    st.divider()

    with tab2:
        st.subheader("Run Benchmark Suite")

        suite_options = {
            "simple_tasks": "Simple Tasks (single tool)",
            "multi_step_tasks": "Multi-Step Tasks (workflows)",
            "edge_cases": "Edge Cases (tricky scenarios)",
        }

        selected_suite = st.selectbox(
            "Select benchmark suite:",
            options=list(suite_options.keys()),
            format_func=lambda x: suite_options[x],
        )

        # Show tasks in suite
        suite = get_builtin_suite(selected_suite)
        if suite:
            with st.expander(f"📋 Tasks in {suite.name} ({len(suite.tasks)})", expanded=False):
                for task in suite.tasks:
                    st.markdown(f"**{task.id}** [{task.difficulty.value}]")
                    st.write(task.task)
                    if task.expected_tools:
                        st.caption(f"Expected: {', '.join(task.expected_tools)}")
                    st.divider()

        if st.button("▶️ Run Suite", type="primary", disabled=not selected_models):
            st.info("Running full benchmark suite... This may take a while.")
            # Implementation similar to tab1 but for all tasks in suite

    with tab3:
        st.subheader("Results History")
        st.info("Results history will be saved here after running benchmarks.")

        # Check for saved results
        results_dir = Path("results")
        if results_dir.exists():
            result_files = list(results_dir.glob("*.json"))
            if result_files:
                for f in sorted(result_files, reverse=True)[:10]:
                    with st.expander(f.stem):
                        with open(f) as fp:
                            st.json(json.load(fp))
            else:
                st.caption("No saved results yet.")
        else:
            st.caption("No saved results yet.")

    # Footer
    st.divider()
    st.caption("Built with Composio Tool Router | [Documentation](https://docs.composio.dev)")


if __name__ == "__main__":
    main()
