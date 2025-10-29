#!/usr/bin/env python3
"""
Streamlit UI for Reviewing Platinum Benchmark Results

Supports both single-model and multi-model reviews.

Launch with:
    streamlit run review_ui.py
"""

import streamlit as st
import json
from pathlib import Path
from typing import Dict, Any, List, Union
import pandas as pd

st.set_page_config(
    page_title="Platinum Benchmark Review",
    page_icon="💎",
    layout="wide"
)


def load_review_file(file_path: Path) -> Dict[str, Any]:
    """Load review JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def save_review_file(file_path: Path, data: Dict[str, Any]) -> None:
    """Save updated review JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def get_review_sources() -> Dict[str, Union[Path, str]]:
    """Get all review files and directories from platinum_review directory.

    Returns:
        Dict mapping display name to Path (file or directory)
    """
    review_dir = Path("platinum_review")
    if not review_dir.exists():
        return {}

    sources = {}

    # Get directories (multi-model runs)
    for item in sorted(review_dir.iterdir(), reverse=True):
        if item.is_dir():
            combined_file = item / "combined_review.json"
            if combined_file.exists():
                sources[f"📁 {item.name}"] = item

    # Get individual JSON files (single-model runs)
    for item in sorted(review_dir.glob("*.json"), reverse=True):
        sources[f"📄 {item.name}"] = item

    return sources


def is_multi_model_review(data: Dict[str, Any]) -> bool:
    """Check if this is a multi-model review based on data structure."""
    # Multi-model reviews have 'models_evaluated' in metadata
    metadata = data.get("metadata", {})
    return "models_evaluated" in metadata


def main():
    st.title("💎 Platinum Benchmark Review Tool")

    # Sidebar - Source selection
    st.sidebar.header("📁 Select Review Source")

    review_sources = get_review_sources()

    if not review_sources:
        st.error("No review files or directories found in `platinum_review/` directory.")
        st.info("Run `create_platinum_simple.py` or `create_platinum_multi.py` first to generate review data.")
        return

    selected_name = st.sidebar.selectbox(
        "Review Source",
        options=list(review_sources.keys())
    )

    selected_source = review_sources[selected_name]

    # Determine if this is a directory or file
    if selected_source.is_dir():
        # Multi-model review
        review_file = selected_source / "combined_review.json"
        is_directory = True
    else:
        # Single-model review
        review_file = selected_source
        is_directory = False

    # Load data
    if 'review_data' not in st.session_state or st.session_state.get('current_file') != review_file:
        st.session_state.review_data = load_review_file(review_file)
        st.session_state.current_file = review_file
        st.session_state.unsaved_changes = False
        st.session_state.is_multi_model = is_multi_model_review(st.session_state.review_data)

    data = st.session_state.review_data
    is_multi_model = st.session_state.is_multi_model

    # Display metadata
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Metadata")
    metadata = data.get("metadata", {})
    st.sidebar.write(f"**Dataset:** {metadata.get('dataset_name', 'N/A')}")

    if is_multi_model:
        # Multi-model metadata
        models = metadata.get('models_evaluated', [])
        st.sidebar.write(f"**Models:** {len(models)}")
        for model in models:
            st.sidebar.write(f"  • {model}")
        st.sidebar.write(f"**Total Questions:** {metadata.get('num_questions', 'N/A')}")

        agreement_stats = metadata.get('agreement_stats', {})
        st.sidebar.write(f"**All Exact Match:** {agreement_stats.get('all_exact_match', 'N/A')} ✅")
        st.sidebar.write(f"**Some Exact Match:** {agreement_stats.get('some_exact_match', 'N/A')} ⚠️")
        st.sidebar.write(f"**No Exact Match:** {agreement_stats.get('all_no_match', 'N/A')} ❌")
        st.sidebar.write(f"**Needs Review:** {agreement_stats.get('num_needs_review', 'N/A')}")
    else:
        # Single-model metadata
        st.sidebar.write(f"**Model:** {metadata.get('model_name', 'N/A')}")
        st.sidebar.write(f"**Total Questions:** {metadata.get('num_questions', 'N/A')}")

        # Show match type stats if available
        if 'num_exact_match' in metadata:
            st.sidebar.write(f"**Exact Match:** {metadata.get('num_exact_match', 'N/A')} ✅")
            st.sidebar.write(f"**Substring Match:** {metadata.get('num_substring_match', 'N/A')} ⚠️")
            st.sidebar.write(f"**No Match:** {metadata.get('num_no_match', 'N/A')} ❌")
            st.sidebar.write(f"**Needs Review:** {metadata.get('num_needs_review', 'N/A')}")

    # Save button
    st.sidebar.markdown("---")
    if st.sidebar.button("💾 Save Changes", type="primary", disabled=not st.session_state.unsaved_changes):
        save_review_file(review_file, st.session_state.review_data)
        st.session_state.unsaved_changes = False
        st.sidebar.success("✅ Saved!")

    if st.session_state.unsaved_changes:
        st.sidebar.warning("⚠️ Unsaved changes")

    # Main content
    questions = data.get("questions", [])

    # Filter controls
    st.header("🔍 Filter & Review")

    if is_multi_model:
        # Multi-model filters
        col1, col2, col3 = st.columns(3)

        with col1:
            filter_status = st.multiselect(
                "Review Status",
                options=["pending", "verified", "revised", "rejected"],
                default=["pending"]
            )

        with col2:
            filter_agreement = st.multiselect(
                "Model Agreement",
                options=["all_exact", "some_exact", "no_exact"],
                default=["some_exact", "no_exact"],
                help="all_exact: All models exact match | some_exact: Some models match | no_exact: No models match"
            )

        with col3:
            show_only_needs_review = st.checkbox(
                "Only show needs review",
                value=True,
                help="Show only questions that need manual review (pending status)"
            )
    else:
        # Single-model filters
        col1, col2, col3 = st.columns(3)

        with col1:
            filter_status = st.multiselect(
                "Review Status",
                options=["pending", "verified", "revised", "rejected"],
                default=["pending"]
            )

        with col2:
            filter_correctness = st.multiselect(
                "Correctness",
                options=["correct", "incorrect"],
                default=["incorrect"]
            )

        with col3:
            show_only_needs_review = st.checkbox(
                "Only show needs review",
                value=True,
                help="Show only questions that need manual review (pending status)"
            )

    # Apply filters
    filtered_questions = []
    for i, q in enumerate(questions):
        q['_index'] = i  # Store original index

        # Filter by status
        if filter_status and q.get('review_status') not in filter_status:
            continue

        # Filter by agreement (multi-model)
        if is_multi_model and filter_agreement:
            if "all_exact" in filter_agreement and not q.get('all_exact_match'):
                if not ("some_exact" in filter_agreement and q.get('num_exact', 0) > 0):
                    if not ("no_exact" in filter_agreement and q.get('num_exact', 0) == 0):
                        continue
            elif "some_exact" in filter_agreement and not (q.get('num_exact', 0) > 0 and not q.get('all_exact_match')):
                if not ("all_exact" in filter_agreement and q.get('all_exact_match')):
                    if not ("no_exact" in filter_agreement and q.get('num_exact', 0) == 0):
                        continue
            elif "no_exact" in filter_agreement and not (q.get('num_exact', 0) == 0):
                if not ("all_exact" in filter_agreement and q.get('all_exact_match')):
                    if not ("some_exact" in filter_agreement and q.get('num_exact', 0) > 0 and not q.get('all_exact_match')):
                        continue

        # Filter by correctness (single-model)
        if not is_multi_model and filter_correctness:
            is_correct = q.get('is_correct', False)
            correctness = "correct" if is_correct else "incorrect"
            if correctness not in filter_correctness:
                continue

        # Filter by needs review
        if show_only_needs_review and q.get('review_status') != 'pending':
            continue

        filtered_questions.append(q)

    st.info(f"Showing {len(filtered_questions)} of {len(questions)} questions")

    # Display questions
    if not filtered_questions:
        st.warning("No questions match the current filters.")
        return

    # Question selector
    st.markdown("---")

    question_selector = st.selectbox(
        "Select Question",
        options=range(len(filtered_questions)),
        format_func=lambda i: f"Q{i+1}: {filtered_questions[i]['question'][:80]}..."
    )

    if question_selector is not None:
        if is_multi_model:
            display_multi_model_question_review(filtered_questions[question_selector])
        else:
            display_single_model_question_review(filtered_questions[question_selector])


def display_single_model_question_review(question: Dict[str, Any]):
    """Display a single-model question for review."""

    original_index = question['_index']

    st.markdown("---")
    st.subheader(f"Question ID: {question.get('question_id', 'N/A')}")

    # Status badge
    status = question.get('review_status', 'pending')
    is_correct = question.get('is_correct', False)
    match_type = question.get('match_type', 'unknown')

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if is_correct:
            st.success("✅ Model answered correctly")
        else:
            st.error("❌ Model answered incorrectly")

    with col2:
        match_type_colors = {
            'exact': '✅',
            'substring': '⚠️',
            'none': '❌',
            'unknown': '⚪'
        }
        st.info(f"{match_type_colors.get(match_type, '⚪')} {match_type.upper()} match")

    with col3:
        status_colors = {
            'pending': '🟡',
            'verified': '🟢',
            'revised': '🔵',
            'rejected': '🔴'
        }
        st.markdown(f"### {status_colors.get(status, '⚪')} {status.upper()}")

    # Question text
    st.markdown("### 📝 Question")
    st.info(question.get('question', 'N/A'))

    # Oracle context
    with st.expander("📚 Oracle Context (Supporting Documents)", expanded=False):
        st.text(question.get('oracle_context', 'N/A'))

    # Answers
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 Gold Answers")
        gold_answers = question.get('gold_answers', [])
        for i, ans in enumerate(gold_answers, 1):
            st.markdown(f"{i}. `{ans}`")

    with col2:
        st.markdown("### 🤖 Model Prediction")
        st.markdown(f"`{question.get('predicted_answer', 'N/A')}`")

    # Full model response
    with st.expander("💬 Full Model Response", expanded=False):
        st.text(question.get('full_response', 'N/A'))

    # Review section
    render_review_section(question, original_index)


def display_multi_model_question_review(question: Dict[str, Any]):
    """Display a multi-model question for review."""

    original_index = question['_index']

    st.markdown("---")
    st.subheader(f"Question ID: {question.get('question_id', 'N/A')}")

    # Status badge
    status = question.get('review_status', 'pending')
    all_exact = question.get('all_exact_match', False)
    all_agree = question.get('all_models_agree', False)
    num_exact = question.get('num_exact', 0)
    num_substring = question.get('num_substring', 0)
    num_no_match = question.get('num_no_match', 0)

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if all_exact:
            st.success(f"✅ All models exact match ({num_exact})")
        elif all_agree:
            st.warning(f"⚠️ All models agree (exact: {num_exact}, substring: {num_substring})")
        else:
            st.error(f"❌ Models disagree (exact: {num_exact}, substring: {num_substring}, none: {num_no_match})")

    with col2:
        agreement_display = f"{num_exact}/{num_exact + num_substring + num_no_match} exact"
        st.info(f"📊 {agreement_display}")

    with col3:
        status_colors = {
            'pending': '🟡',
            'verified': '🟢',
            'revised': '🔵',
            'rejected': '🔴'
        }
        st.markdown(f"### {status_colors.get(status, '⚪')} {status.upper()}")

    # Question text
    st.markdown("### 📝 Question")
    st.info(question.get('question', 'N/A'))

    # Oracle context
    with st.expander("📚 Oracle Context (Supporting Documents)", expanded=False):
        st.text(question.get('oracle_context', 'N/A'))

    # Model predictions table
    st.markdown("### 🤖 Model Predictions")

    model_predictions = question.get('model_predictions', {})
    match_types = question.get('match_types', {})
    gold_answers = question.get('gold_answers', [])

    # Create DataFrame for display
    pred_data = []
    for model, prediction in model_predictions.items():
        match_type = match_types.get(model, 'unknown')
        match_icon = {'exact': '✅', 'substring': '⚠️', 'none': '❌'}.get(match_type, '⚪')

        pred_data.append({
            'Model': model,
            'Prediction': prediction,
            'Match': f"{match_icon} {match_type}"
        })

    df = pd.DataFrame(pred_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Gold answers
    st.markdown("### 🎯 Gold Answers")
    for i, ans in enumerate(gold_answers, 1):
        st.markdown(f"{i}. `{ans}`")

    # Full model responses
    with st.expander("💬 Full Model Responses", expanded=False):
        model_responses = question.get('model_full_responses', {})
        for model, response in model_responses.items():
            st.markdown(f"**{model}:**")
            st.text(response)
            st.markdown("---")

    # Review section
    render_review_section(question, original_index)


def render_review_section(question: Dict[str, Any], original_index: int):
    """Render the review section (common for both single and multi-model)."""

    st.markdown("---")
    st.markdown("## ✏️ Review & Annotation")

    status = question.get('review_status', 'pending')

    col1, col2 = st.columns([1, 2])

    with col1:
        # Review status dropdown
        new_status = st.selectbox(
            "Review Status",
            options=["pending", "verified", "revised", "rejected"],
            index=["pending", "verified", "revised", "rejected"].index(status),
            key=f"status_{original_index}",
            help="""
            - **verified**: Gold answer is correct, model(s) made an error
            - **revised**: Model answer(s) are valid alternatives
            - **rejected**: Question is unanswerable/ambiguous/gold is wrong
            """
        )

        # Update if changed
        if new_status != status:
            st.session_state.review_data['questions'][original_index]['review_status'] = new_status
            st.session_state.unsaved_changes = True

    with col2:
        # Platinum target (only show if not rejected)
        if new_status != 'rejected':
            st.markdown("**Platinum Target (all acceptable answers)**")
            st.caption("Enter comma-separated answers")

            current_platinum = question.get('platinum_target')
            if current_platinum is None:
                default_value = ", ".join(question.get('gold_answers', []))
            elif isinstance(current_platinum, list):
                default_value = ", ".join(current_platinum)
            else:
                default_value = str(current_platinum)

            platinum_input = st.text_area(
                "Platinum Target",
                value=default_value,
                key=f"platinum_{original_index}",
                label_visibility="collapsed",
                height=100
            )

            # Parse into list
            new_platinum = [ans.strip() for ans in platinum_input.split(',') if ans.strip()]

            # Update if changed
            if new_platinum != question.get('platinum_target'):
                st.session_state.review_data['questions'][original_index]['platinum_target'] = new_platinum
                st.session_state.unsaved_changes = True
        else:
            st.info("Question rejected - platinum_target set to null")
            if question.get('platinum_target') is not None:
                st.session_state.review_data['questions'][original_index]['platinum_target'] = None
                st.session_state.unsaved_changes = True

    # Review notes
    st.markdown("**Review Notes**")
    current_notes = question.get('review_notes', '')
    new_notes = st.text_area(
        "Notes",
        value=current_notes,
        key=f"notes_{original_index}",
        label_visibility="collapsed",
        height=80,
        placeholder="Add any notes about why you made this decision..."
    )

    if new_notes != current_notes:
        st.session_state.review_data['questions'][original_index]['review_notes'] = new_notes
        st.session_state.unsaved_changes = True

    # Helper info
    with st.expander("ℹ️ Review Guidelines", expanded=False):
        st.markdown("""
        ### Decision Framework

        **Ask these questions in order:**

        1. **Can I answer this from the oracle context alone?**
           - If NO → Mark as `rejected`

        2. **Does my answer match the gold label?**
           - If NO → Mark as `rejected` (or `revised` if fixable)

        3. **Are the model predictions semantically equivalent to gold?**
           - If YES → Mark as `revised` and add predictions to platinum_target
           - If NO → Mark as `verified` (models are wrong)

        ### Multi-Model Strategy

        - **All exact match**: Already auto-verified, low priority
        - **Some exact match**: Review to see if other models found valid alternatives
        - **No exact match**: High priority - likely problem with question/answer

        ### Examples

        - **verified**: Gold is "1755", all models say "1940" → Models made same error
        - **revised**: Gold is "40 members", some models say "40" → Add both to platinum_target
        - **rejected**: Question is ambiguous or unanswerable from context
        """)


if __name__ == "__main__":
    main()
