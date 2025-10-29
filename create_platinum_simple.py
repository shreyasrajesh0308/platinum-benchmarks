#!/usr/bin/env python3
"""
Create Platinum Benchmark - Simple Single Model Version

Start with one model, evaluate on supporting documents only, generate review file.

Usage:
    python create_platinum_simple.py --dataset 2wikimultihopqa --model gpt-4o --num-questions 100
    python create_platinum_simple.py --dataset musique --model gpt-4o-mini --num-questions 50
"""

import json
import re
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


class PlatinumBenchmarkCreator:
    """Create platinum benchmark using supporting documents only."""

    def __init__(self, dataset_name: str, model_name: str,
                 num_questions: int = None, dataset_dir: str = None):
        """Initialize the creator.

        Args:
            dataset_name: '2wikimultihopqa' or 'musique'
            model_name: Model to evaluate (e.g., 'gpt-4o')
            num_questions: Number of questions (None for all)
            dataset_dir: Directory with dataset files
        """
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.num_questions = num_questions

        # Set dataset path
        if dataset_dir is None:
            dataset_dir = "/home/shreyas/NLP/SM/gensemworkspaces/HippoRAG/reproduce/dataset"

        self.dataset_file = Path(dataset_dir) / f"{dataset_name}.json"

        console.print(f"[bold cyan]Creating Platinum Benchmark[/bold cyan]")
        console.print(f"Dataset: [cyan]{self.dataset_name}[/cyan]")
        console.print(f"Model: [cyan]{self.model_name}[/cyan]")
        console.print(f"Questions: [cyan]{num_questions or 'all'}[/cyan]")

        if not self.dataset_file.exists():
            console.print(f"[red]✗ Dataset not found: {self.dataset_file}[/red]")
            import sys
            sys.exit(1)

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load questions with supporting context only."""
        console.print(f"[cyan]Loading dataset...[/cyan]")

        with open(self.dataset_file, 'r') as f:
            data = json.load(f)

        questions_data = []
        for i, item in enumerate(data):
            if self.num_questions and i >= self.num_questions:
                break

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
        """Format supporting facts as oracle context (supporting_only=True).

        Args:
            context: List of [title, sentences] pairs
            supporting_facts: List of [title, sentence_idx] pairs

        Returns:
            Formatted context string with only supporting sentences
        """
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

    def run_model(self, questions_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run model on all questions."""
        console.print(f"[cyan]Running {self.model_name}...[/cyan]")

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
        generator = PlatinumAnswerGenerator(
            model_name="gemini-2.5-flash-lite",
            backend="openai",
            generation_params={"temperature": 0.0}
        )

        # Run inference
        with Progress(
            SpinnerColumn(),
            TextColumn(f"[cyan]Processing {len(model_inputs)} questions..."),
            console=console
        ) as progress:
            task = progress.add_task("Running...", total=None)

            try:
                results_dataset = generator(model_inputs)
                progress.update(task, completed=100, total=100)
            except Exception as e:
                console.print(f"[red]✗ Model failed: {e}[/red]")
                import sys
                sys.exit(1)

        results = list(results_dataset.dataset)
        console.print(f"[green]✓ Completed ({len(results)} results)[/green]")

        return results

    def analyze_results(self, questions_data: List[Dict[str, Any]],
                       results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze results and create review data."""
        console.print(f"[cyan]Analyzing results...[/cyan]")

        review_data = []

        for qdata, result in zip(questions_data, results):
            predicted = result["predicted_answer"]
            gold_answers = qdata["gold_answers"]

            # Check match type: exact, substring, or none
            match_type, is_correct = self._check_answer_match(predicted, gold_answers)

            # Automatically set status based on match type
            if match_type == "exact":
                # Exact match - no review needed
                review_status = "verified"
                platinum_target = gold_answers
            else:
                # Substring or no match - needs manual review
                review_status = "pending"
                platinum_target = None

            review_item = {
                "question_id": qdata["question_id"],
                "question": qdata["question"],
                "gold_answers": gold_answers,
                "predicted_answer": predicted,
                "full_response": result["full_response"],
                "oracle_context": qdata["oracle_context"],
                "supporting_facts": qdata["supporting_facts"],
                "is_correct": is_correct,
                "match_type": match_type,  # exact, substring, or none

                # Fields for manual review
                "review_status": review_status,
                "platinum_target": platinum_target,
                "review_notes": ""
            }

            review_data.append(review_item)

        # Calculate stats
        num_exact = sum(1 for item in review_data if item["match_type"] == "exact")
        num_substring = sum(1 for item in review_data if item["match_type"] == "substring")
        num_no_match = sum(1 for item in review_data if item["match_type"] == "none")

        console.print(f"[green]Exact match (auto-verified): {num_exact}/{len(review_data)} ({num_exact/len(review_data)*100:.1f}%)[/green]")
        console.print(f"[yellow]Substring match (needs review): {num_substring}/{len(review_data)} ({num_substring/len(review_data)*100:.1f}%)[/yellow]")
        console.print(f"[red]No match (needs review): {num_no_match}/{len(review_data)} ({num_no_match/len(review_data)*100:.1f}%)[/red]")

        return review_data

    def _check_answer_match(self, predicted: str, gold_answers: List[str]) -> tuple[str, bool]:
        """Check answer match and return match type.

        Args:
            predicted: Predicted answer
            gold_answers: List of gold answers

        Returns:
            Tuple of (match_type, is_correct)
            match_type: "exact", "substring", or "none"
            is_correct: True if exact or substring match
        """
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

    def save_review_file(self, review_data: List[Dict[str, Any]]) -> Path:
        """Save review data to JSON file."""
        output_dir = Path("platinum_review")
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f"{self.dataset_name}_{self.model_name.replace('/', '_')}_{timestamp}.json"

        # Calculate stats
        num_exact = sum(1 for item in review_data if item["match_type"] == "exact")
        num_substring = sum(1 for item in review_data if item["match_type"] == "substring")
        num_no_match = sum(1 for item in review_data if item["match_type"] == "none")

        output_data = {
            "metadata": {
                "dataset_name": self.dataset_name,
                "model_name": self.model_name,
                "num_questions": len(review_data),
                "timestamp": datetime.now().isoformat(),
                "num_exact_match": num_exact,
                "num_substring_match": num_substring,
                "num_no_match": num_no_match,
                "num_auto_verified": num_exact,
                "num_needs_review": num_substring + num_no_match
            },
            "review_instructions": {
                "status_options": ["verified", "revised", "rejected", "pending"],
                "verified": "Gold answer is correct, model made an error",
                "revised": "Need to add alternative valid answers to platinum_target",
                "rejected": "Question is unanswerable/ambiguous/gold is wrong",
                "platinum_target_note": "List of all acceptable answers (null for rejected)"
            },
            "questions": review_data
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        console.print(f"[green]✓ Review file saved: {output_file}[/green]")
        return output_file

    def display_summary(self, review_data: List[Dict[str, Any]]) -> None:
        """Display summary."""
        console.print("\n" + "="*70)
        console.print("[bold cyan]Platinum Benchmark Summary[/bold cyan]")
        console.print("="*70)

        total = len(review_data)
        num_exact = sum(1 for item in review_data if item["match_type"] == "exact")
        num_substring = sum(1 for item in review_data if item["match_type"] == "substring")
        num_no_match = sum(1 for item in review_data if item["match_type"] == "none")
        num_needs_review = num_substring + num_no_match

        table = Table(title="Match Analysis")
        table.add_column("Match Type", style="cyan")
        table.add_column("Count", style="green")
        table.add_column("Percentage", style="yellow")
        table.add_column("Status", style="magenta")

        table.add_row(
            "Exact match",
            str(num_exact),
            f"{num_exact/total*100:.1f}%",
            "✅ Auto-verified"
        )
        table.add_row(
            "Substring match",
            str(num_substring),
            f"{num_substring/total*100:.1f}%",
            "⚠️ Needs review"
        )
        table.add_row(
            "No match",
            str(num_no_match),
            f"{num_no_match/total*100:.1f}%",
            "❌ Needs review"
        )

        console.print(table)

        console.print(f"\n[bold]Review Required: {num_needs_review}/{total} questions[/bold]")

        console.print(f"\n[bold yellow]Next Steps:[/bold yellow]")
        console.print(f"1. Open Streamlit UI: [cyan]streamlit run review_ui.py[/cyan]")
        console.print(f"2. Review {num_needs_review} questions with 'pending' status")
        console.print(f"3. For each pending question, determine:")
        console.print(f"   - [green]verified[/green]: Gold is correct, model wrong")
        console.print(f"   - [blue]revised[/blue]: Model answer is valid alternative, add to platinum_target")
        console.print(f"   - [red]rejected[/red]: Question is unanswerable/ambiguous")
        console.print(f"4. Save your changes in the UI")

    def run(self):
        """Run complete pipeline."""
        # Load data
        questions_data = self.load_dataset()

        # Run model
        results = self.run_model(questions_data)

        # Analyze
        review_data = self.analyze_results(questions_data, results)

        # Save
        output_file = self.save_review_file(review_data)

        # Summary
        self.display_summary(review_data)

        console.print(f"\n[bold green]✅ Done! Review file: {output_file}[/bold green]")


def main():
    parser = argparse.ArgumentParser(description="Create platinum benchmark (single model)")
    parser.add_argument("--dataset", required=True,
                       choices=["2wikimultihopqa", "musique"],
                       help="Dataset name")
    parser.add_argument("--model", required=True,
                       help="Model name (e.g., gpt-4o, gpt-4o-mini)")
    parser.add_argument("--num-questions", type=int, default=None,
                       help="Number of questions (default: all)")
    parser.add_argument("--dataset-dir", default=None,
                       help="Dataset directory")

    args = parser.parse_args()

    creator = PlatinumBenchmarkCreator(
        dataset_name=args.dataset,
        model_name=args.model,
        num_questions=args.num_questions,
        dataset_dir=args.dataset_dir
    )

    creator.run()


if __name__ == "__main__":
    main()
