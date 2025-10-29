#!/usr/bin/env python3
"""
Create Platinum Benchmark - Multi-Model Version

Run multiple models and aggregate results for consensus-based verification.

Usage:
    python create_platinum_multi.py --dataset 2wikimultihopqa --models gpt-4o gpt-4o-mini --num-questions 50
    python create_platinum_multi.py --dataset musique --models gpt-4o claude-3-5-sonnet-20241022 gemini-2.0-flash-exp
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

try:
    from bespokelabs import curator
    CURATOR_AVAILABLE = True
except ImportError:
    print("Error: Curator not available. Install with: pip install bespokelabs-curator")
    import sys
    sys.exit(1)

console = Console()


class PlatinumAnswerGenerator(curator.LLM):
    """Curator class for generating answers with supporting documents only."""

    return_completions_object = True

    def prompt(self, input_data):
        """Create a prompt with supporting documents (oracle context)."""
        evidence_text = input_data['oracle_context']
        question = input_data['question']

        prompt_text = f"""{evidence_text}

Question: {question}

Thought: """

        # One-shot example (same as oracle_eval.py)
        one_shot_docs = (
            """Wikipedia Title: The Last Horse\nThe Last Horse (Spanish:El último caballo) is a 1950 Spanish comedy film directed by Edgar Neville starring Fernando Fernán Gómez.\n"""
            """Wikipedia Title: Southampton\nThe University of Southampton, which was founded in 1862 and received its Royal Charter as a university in 1952, has over 22,000 students. The university is ranked in the top 100 research universities in the world in the Academic Ranking of World Universities 2010. In 2010, the THES - QS World University Rankings positioned the University of Southampton in the top 80 universities in the world. The university considers itself one of the top 5 research universities in the UK. The university has a global reputation for research into engineering sciences, oceanography, chemistry, cancer sciences, sound and vibration research, computer science and electronics, optoelectronics and textile conservation at the Textile Conservation Centre (which is due to close in October 2009.) It is also home to the National Oceanography Centre, Southampton (NOCS), the focus of Natural Environment Research Council-funded marine research.\n"""
            """Wikipedia Title: Neville A. Stanton\nNeville A. Stanton is a British Professor of Human Factors and Ergonomics at the University of Southampton. Prof Stanton is a Chartered Engineer (C.Eng), Chartered Psychologist (C.Psychol) and Chartered Ergonomist (C.ErgHF). He has written and edited over a forty books and over three hundered peer-reviewed journal papers on applications of the subject. Stanton is a Fellow of the British Psychological Society, a Fellow of The Institute of Ergonomics and Human Factors and a member of the Institution of Engineering and Technology. He has been published in academic journals including "Nature". He has also helped organisations design new human-machine interfaces, such as the Adaptive Cruise Control system for Jaguar Cars.\n"""
        )

        system_prompt = (
            'As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. '
            'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
            'Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations.'
        )

        one_shot_input = (
            f"{one_shot_docs}"
            "\n\nQuestion: When was Neville A. Stanton's employer founded?"
            '\nThought: '
        )

        one_shot_output = (
            "The employer of Neville A. Stanton is University of Southampton. The University of Southampton was founded in 1862. "
            "\nAnswer: 1862."
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": one_shot_input},
            {"role": "assistant", "content": one_shot_output},
            {"role": "user", "content": prompt_text}
        ]

    def parse(self, input_data, response):
        """Parse the answer from the response."""
        answer_text = response["choices"][0]["message"]["content"]

        # Extract final answer after "Answer: "
        if 'Answer: ' in answer_text:
            final_answer = answer_text.split('Answer: ')[-1].strip()
        else:
            final_answer = answer_text.strip()

        # Strip trailing punctuation (., !, ?, etc.)
        final_answer = final_answer.rstrip('.!?,;:')

        return [{
            "question_id": input_data['question_id'],
            "question": input_data['question'],
            "predicted_answer": final_answer,
            "full_response": answer_text,
            "gold_answers": input_data['gold_answers'],
            "oracle_context": input_data['oracle_context']
        }]


class MultiModelPlatinumCreator:
    """Create platinum benchmark with multiple models."""

    def __init__(self, dataset_name: str, model_names: List[str],
                 num_questions: int = None, question_start: int = 0, dataset_dir: str = None):
        """Initialize the multi-model creator.

        Args:
            dataset_name: '2wikimultihopqa' or 'musique'
            model_names: List of model names to evaluate
            num_questions: Number of questions (None for all)
            dataset_dir: Directory with dataset files
        """
        self.dataset_name = dataset_name
        self.model_names = model_names
        self.num_questions = num_questions
        self.question_start = question_start
        # Set dataset path
        if dataset_dir is None:
            dataset_dir = "/home/shreyas/NLP/SM/gensemworkspaces/HippoRAG/reproduce/dataset"

        self.dataset_file = Path(dataset_dir) / f"{dataset_name}.json"

        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = Path("platinum_review") / f"{dataset_name}_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        console.print(f"[bold cyan]Creating Multi-Model Platinum Benchmark[/bold cyan]")
        console.print(f"Dataset: [cyan]{self.dataset_name}[/cyan]")
        console.print(f"Question start: [cyan]{self.question_start}[/cyan]")
        console.print(f"Models: [cyan]{', '.join(self.model_names)}[/cyan]")
        console.print(f"Questions: [cyan]{num_questions or 'all'}[/cyan]")
        console.print(f"Output: [cyan]{self.output_dir}[/cyan]")

        if not self.dataset_file.exists():
            console.print(f"[red]✗ Dataset not found: {self.dataset_file}[/red]")
            import sys
            sys.exit(1)

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load questions with supporting context only."""
        console.print(f"[cyan]Loading dataset...[/cyan]")
        console.print(f"Question start: [cyan]{self.question_start}[/cyan]")
        with open(self.dataset_file, 'r') as f:
            data = json.load(f)

        questions_data = []
        for i, item in enumerate(data):
            if self.num_questions and i >= self.num_questions:
                break

            if i < self.question_start:
                continue

            question_id = item.get("_id", f"q_{i}")
            question = item["question"]
            gold_answers = item.get("answer", [])
            context = item.get("context", [])
            supporting_facts = item.get("supporting_facts", [])

            # Ensure gold_answers is a list
            if isinstance(gold_answers, str):
                gold_answers = [gold_answers]

            # Format ONLY supporting facts as oracle context
            oracle_context = self._format_supporting_context(context, supporting_facts)

            questions_data.append({
                "question_id": question_id,
                "question": question,
                "gold_answers": gold_answers,
                "oracle_context": oracle_context,
                "supporting_facts": supporting_facts
            })

        console.print(f"[green]✓ Loaded {len(questions_data)} questions[/green]")
        return questions_data

    def _format_supporting_context(self, context: List[List[Any]],
                                   supporting_facts: List[List[Any]]) -> str:
        """Format supporting facts as oracle context (supporting_only=True)."""
        # Create mapping from title to sentences
        context_map = {}
        for doc_data in context:
            if len(doc_data) != 2:
                continue
            title, sentences = doc_data
            context_map[title] = sentences

        formatted_docs = []
        for fact in supporting_facts:
            if len(fact) != 2:
                continue

            title, sentence_idx = fact
            if title not in context_map:
                continue

            sentences = context_map[title]
            if isinstance(sentences, list) and 0 <= sentence_idx < len(sentences):
                sentence_text = sentences[sentence_idx]
            elif isinstance(sentences, list):
                sentence_text = " ".join(sentences)
            else:
                sentence_text = str(sentences)

            # Format with title
            formatted_doc = f"Wikipedia Title: {title}\n{sentence_text}"
            formatted_docs.append(formatted_doc)

        return "\n".join(formatted_docs)

    def run_model(self, questions_data: List[Dict[str, Any]], model_name: str) -> List[Dict[str, Any]]:
        """Run a single model on all questions."""
        console.print(f"\n[cyan]Running {model_name}...[/cyan]")

        # Prepare inputs
        model_inputs = []
        for qdata in questions_data:
            model_inputs.append({
                "question_id": qdata["question_id"],
                "question": qdata["question"],
                "oracle_context": qdata["oracle_context"],
                "gold_answers": qdata["gold_answers"]
            })

        # Initialize generator
        if "gpt" in model_name:
            generator = PlatinumAnswerGenerator(
                model_name=model_name,
                generation_params={"temperature": 0.0}
            )
        else:
            generator = PlatinumAnswerGenerator(
                model_name=model_name,
                backend="litellm",
                generation_params={"temperature": 0.0},
                backend_params={
                    "max_requests_per_minute": 2_000,
                    "max_tokens_per_minute": 4_000_000
                }
            )

        # Run inference
        with Progress(
            SpinnerColumn(),
            TextColumn(f"[cyan]Processing {len(model_inputs)} questions with {model_name}..."),
            console=console
        ) as progress:
            task = progress.add_task("Running...", total=None)

            try:
                results_dataset = generator(model_inputs)
                progress.update(task, completed=100, total=100)
            except Exception as e:
                console.print(f"[red]✗ Model {model_name} failed: {e}[/red]")
                return []

        results = list(results_dataset.dataset)
        console.print(f"[green]✓ {model_name} completed ({len(results)} results)[/green]")

        return results

    def _check_answer_match(self, predicted: str, gold_answers: List[str]) -> tuple[str, bool]:
        """Check answer match and return match type."""
        predicted_lower = predicted.lower().strip()

        for gold in gold_answers:
            gold_lower = gold.lower().strip()

            # Exact match
            if predicted_lower == gold_lower:
                return ("exact", True)

        # Check substring match
        for gold in gold_answers:
            gold_lower = gold.lower().strip()

            # Substring match (either direction)
            if gold_lower in predicted_lower or predicted_lower in gold_lower:
                return ("substring", True)

        # No match
        return ("none", False)

    def aggregate_results(self, questions_data: List[Dict[str, Any]],
                         all_model_results: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Aggregate results from all models into combined review format."""
        console.print(f"\n[cyan]Aggregating results from {len(self.model_names)} models...[/cyan]")

        combined_data = []

        for i, qdata in enumerate(questions_data):
            question_id = qdata["question_id"]
            gold_answers = qdata["gold_answers"]

            # Collect predictions and match types from all models
            model_predictions = {}
            model_full_responses = {}
            match_types = {}

            for model_name in self.model_names:
                if model_name in all_model_results and i < len(all_model_results[model_name]):
                    result = all_model_results[model_name][i]
                    predicted = result["predicted_answer"]
                    full_response = result["full_response"]

                    model_predictions[model_name] = predicted
                    model_full_responses[model_name] = full_response

                    # Check match type
                    match_type, _ = self._check_answer_match(predicted, gold_answers)
                    match_types[model_name] = match_type

            # Calculate agreement metrics
            num_exact = sum(1 for mt in match_types.values() if mt == "exact")
            num_substring = sum(1 for mt in match_types.values() if mt == "substring")
            num_no_match = sum(1 for mt in match_types.values() if mt == "none")

            all_exact_match = (num_exact == len(self.model_names))
            all_models_agree = (num_exact + num_substring == len(self.model_names))

            # Auto-verify logic: only if ALL models have exact match
            if all_exact_match:
                review_status = "verified"
                platinum_target = gold_answers
            else:
                review_status = "pending"
                platinum_target = None

            combined_item = {
                "question_id": question_id,
                "question": qdata["question"],
                "gold_answers": gold_answers,
                "oracle_context": qdata["oracle_context"],
                "supporting_facts": qdata["supporting_facts"],

                # Multi-model predictions
                "model_predictions": model_predictions,
                "model_full_responses": model_full_responses,
                "match_types": match_types,

                # Agreement metrics
                "all_models_agree": all_models_agree,
                "all_exact_match": all_exact_match,
                "num_exact": num_exact,
                "num_substring": num_substring,
                "num_no_match": num_no_match,

                # Review fields
                "review_status": review_status,
                "platinum_target": platinum_target,
                "review_notes": ""
            }

            combined_data.append(combined_item)

        # Calculate stats
        num_all_exact = sum(1 for item in combined_data if item["all_exact_match"])
        num_some_exact = sum(1 for item in combined_data if item["num_exact"] > 0 and not item["all_exact_match"])
        num_all_wrong = sum(1 for item in combined_data if item["num_exact"] == 0)

        console.print(f"[green]All exact match (auto-verified): {num_all_exact}/{len(combined_data)}[/green]")
        console.print(f"[yellow]Some exact match (needs review): {num_some_exact}/{len(combined_data)}[/yellow]")
        console.print(f"[red]No exact match (needs review): {num_all_wrong}/{len(combined_data)}[/red]")

        return combined_data

    def save_results(self, questions_data: List[Dict[str, Any]],
                    all_model_results: Dict[str, List[Dict[str, Any]]],
                    combined_data: List[Dict[str, Any]]) -> None:
        """Save individual model results and combined review file."""
        console.print(f"\n[cyan]Saving results...[/cyan]")

        # Save individual model results
        for model_name, results in all_model_results.items():
            model_file = self.output_dir / f"{model_name.replace('/', '_')}.json"

            # Calculate model-specific stats
            num_exact = sum(1 for r, q in zip(results, questions_data)
                          if self._check_answer_match(r["predicted_answer"], q["gold_answers"])[0] == "exact")
            num_substring = sum(1 for r, q in zip(results, questions_data)
                              if self._check_answer_match(r["predicted_answer"], q["gold_answers"])[0] == "substring")
            num_no_match = sum(1 for r, q in zip(results, questions_data)
                             if self._check_answer_match(r["predicted_answer"], q["gold_answers"])[0] == "none")

            model_data = {
                "metadata": {
                    "model_name": model_name,
                    "dataset_name": self.dataset_name,
                    "num_questions": len(results),
                    "num_exact_match": num_exact,
                    "num_substring_match": num_substring,
                    "num_no_match": num_no_match
                },
                "results": results
            }

            with open(model_file, 'w') as f:
                json.dump(model_data, f, indent=2)

            console.print(f"[green]✓ Saved {model_name} results to {model_file.name}[/green]")

        # Save combined review file
        combined_file = self.output_dir / "combined_review.json"

        num_all_exact = sum(1 for item in combined_data if item["all_exact_match"])
        num_some_exact = sum(1 for item in combined_data if item["num_exact"] > 0 and not item["all_exact_match"])
        num_all_wrong = sum(1 for item in combined_data if item["num_exact"] == 0)

        combined_output = {
            "metadata": {
                "dataset_name": self.dataset_name,
                "models_evaluated": self.model_names,
                "num_questions": len(combined_data),
                "timestamp": datetime.now().isoformat(),
                "agreement_stats": {
                    "all_exact_match": num_all_exact,
                    "some_exact_match": num_some_exact,
                    "all_no_match": num_all_wrong,
                    "num_auto_verified": num_all_exact,
                    "num_needs_review": num_some_exact + num_all_wrong
                }
            },
            "review_instructions": {
                "status_options": ["verified", "revised", "rejected", "pending"],
                "verified": "Gold answer is correct, all models agreed or made same error",
                "revised": "Some model answers are valid alternatives, add to platinum_target",
                "rejected": "Question is unanswerable/ambiguous/gold is wrong",
                "platinum_target_note": "List of all acceptable answers (null for rejected)"
            },
            "questions": combined_data
        }

        with open(combined_file, 'w') as f:
            json.dump(combined_output, f, indent=2)

        console.print(f"[green]✓ Saved combined review to {combined_file.name}[/green]")

        # Save metadata summary
        metadata_file = self.output_dir / "metadata.json"
        metadata = {
            "dataset_name": self.dataset_name,
            "models": self.model_names,
            "num_questions": len(combined_data),
            "timestamp": datetime.now().isoformat(),
            "output_directory": str(self.output_dir),
            "agreement_stats": combined_output["metadata"]["agreement_stats"]
        }

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        console.print(f"[green]✓ Saved metadata to {metadata_file.name}[/green]")

    def display_summary(self, combined_data: List[Dict[str, Any]]) -> None:
        """Display summary statistics."""
        console.print("\n" + "="*70)
        console.print("[bold cyan]Multi-Model Platinum Benchmark Summary[/bold cyan]")
        console.print("="*70)

        total = len(combined_data)
        num_all_exact = sum(1 for item in combined_data if item["all_exact_match"])
        num_some_exact = sum(1 for item in combined_data if item["num_exact"] > 0 and not item["all_exact_match"])
        num_all_wrong = sum(1 for item in combined_data if item["num_exact"] == 0)
        num_needs_review = num_some_exact + num_all_wrong

        table = Table(title="Agreement Analysis")
        table.add_column("Category", style="cyan")
        table.add_column("Count", style="green")
        table.add_column("Percentage", style="yellow")
        table.add_column("Status", style="magenta")

        table.add_row(
            "All models exact match",
            str(num_all_exact),
            f"{num_all_exact/total*100:.1f}%",
            "✅ Auto-verified"
        )
        table.add_row(
            "Some models exact match",
            str(num_some_exact),
            f"{num_some_exact/total*100:.1f}%",
            "⚠️ Needs review"
        )
        table.add_row(
            "No exact matches",
            str(num_all_wrong),
            f"{num_all_wrong/total*100:.1f}%",
            "❌ Needs review"
        )

        console.print(table)

        console.print(f"\n[bold]Review Required: {num_needs_review}/{total} questions[/bold]")
        console.print(f"[bold]Output Directory: {self.output_dir}[/bold]")

        console.print(f"\n[bold yellow]Next Steps:[/bold yellow]")
        console.print(f"1. Open Streamlit UI: [cyan]streamlit run review_ui.py[/cyan]")
        console.print(f"2. Select the directory: [cyan]{self.output_dir.name}[/cyan]")
        console.print(f"3. Review {num_needs_review} questions with 'pending' status")
        console.print(f"4. Focus on questions where models disagree")

    def run(self):
        """Run complete multi-model pipeline."""
        # Load dataset
        questions_data = self.load_dataset()

        # Run all models
        all_model_results = {}
        for model_name in self.model_names:
            results = self.run_model(questions_data, model_name)
            if results:
                all_model_results[model_name] = results

        if not all_model_results:
            console.print("[red]✗ No models completed successfully![/red]")
            return

        # Aggregate results
        combined_data = self.aggregate_results(questions_data, all_model_results)

        # Save all results
        self.save_results(questions_data, all_model_results, combined_data)

        # Display summary
        self.display_summary(combined_data)

        console.print(f"\n[bold green]✅ Multi-model evaluation complete![/bold green]")


def main():
    parser = argparse.ArgumentParser(description="Create platinum benchmark with multiple models")
    parser.add_argument("--dataset", required=True,
                       choices=["2wikimultihopqa", "musique"],
                       help="Dataset name")
    parser.add_argument("--models", required=True, nargs="+",
                       help="List of model names (e.g., gpt-4o gpt-4o-mini)")
    parser.add_argument("--num-questions", type=int, default=None,
                       help="Number of questions (default: all)")
    parser.add_argument("--question-start", type=int, default=0,
                       help="Question start index")
    parser.add_argument("--dataset-dir", default=None,
                       help="Dataset directory")

    args = parser.parse_args()

    creator = MultiModelPlatinumCreator(
        dataset_name=args.dataset,
        model_names=args.models,
        num_questions=args.num_questions,
        question_start=args.question_start,
        dataset_dir=args.dataset_dir
    )

    creator.run()


if __name__ == "__main__":
    main()
