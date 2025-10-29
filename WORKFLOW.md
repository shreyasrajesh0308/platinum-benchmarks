# Platinum Benchmark Creation Workflow

## Overview

This workflow helps you create platinum benchmarks for multihop QA datasets (2WikiMultihopQA and Musique) using automated matching and manual review.

## Step 1: Generate Initial Results

Run the evaluation script with one model:

```bash
# Test with 50 questions from 2WikiMultihopQA
python create_platinum_simple.py --dataset 2wikimultihopqa --model gpt-4o --num-questions 50

# Or test with Musique
python create_platinum_simple.py --dataset musique --model gpt-4o-mini --num-questions 100
```

### What happens:
- Loads questions with **supporting documents only** (oracle context)
- Runs the model on all questions
- Automatically categorizes answers by match type:
  - **Exact match** → Auto-verified ✅ (no review needed)
  - **Substring match** → Pending ⚠️ (needs review)
  - **No match** → Pending ❌ (needs review)

### Output:
```
platinum_review/2wikimultihopqa_gpt-4o_20250421_143022.json
```

## Step 2: Review in Streamlit UI

Launch the review interface:

```bash
streamlit run review_ui.py
```

### UI Features:
- **Auto-filtering**: Shows only pending questions by default
- **Match type display**: See exact/substring/none for each question
- **Side-by-side comparison**: Gold answers vs Model predictions
- **Oracle context**: View supporting documents used
- **Quick annotation**: Update status and platinum_target
- **Auto-save**: Tracks unsaved changes

## Step 3: Review Each Question

For each pending question, follow this decision tree:

### Decision Framework

1. **Read the oracle context and question**
   - Can you answer it from the context alone?
   - If NO → Mark as `rejected`

2. **Answer it yourself (don't look at gold yet)**
   - What's your answer from the context?

3. **Compare your answer to gold**
   - Does it match?
   - If NO → Mark as `rejected` (gold is wrong)

4. **Look at the model's prediction**
   - Is it semantically equivalent to gold?
   - If YES → Mark as `revised`, add both to `platinum_target`
   - If NO → Mark as `verified` (model is wrong, gold is correct)

### Status Options

| Status | When to Use | Platinum Target |
|--------|-------------|-----------------|
| **verified** | Gold is correct, model made an error | `gold_answers` |
| **revised** | Model's answer is a valid alternative | `gold_answers + model_answer + other_variants` |
| **rejected** | Question is unanswerable/ambiguous/gold is wrong | `null` |
| **pending** | Not yet reviewed | `null` |

## Step 4: Examples

### Example 1: Substring Match → Revised
```
Question: "How many members are in the Florida Senate?"
Gold: ["40 members"]
Model: "40"
Match type: substring

Decision: REVISED
Reason: "40" is semantically correct, just missing "members"
Platinum target: ["40 members", "40", "forty", "forty members"]
```

### Example 2: No Match → Verified
```
Question: "When was Moscow State University founded?"
Gold: ["1755"]
Model: "1940"
Match type: none

Decision: VERIFIED
Reason: Gold is correct (1755), model confused the renaming date (1940)
Platinum target: ["1755"]
```

### Example 3: No Match → Rejected
```
Question: "In what year was the university where Sergei Aleksandrovich Tokarev was a professor founded?"
Gold: ["1755"]
Model: "1993"
Match type: none

Decision: REJECTED
Reason: Context has 8 people named "Sergei Aleksandrovich [X]" - ambiguous reference
Platinum target: null
```

## Step 5: Save and Iterate

1. Click **"💾 Save Changes"** in the UI
2. Updated JSON file is saved back
3. Later: Run more models and merge results
4. Continue reviewing failures from new models

## Statistics Tracking

The system tracks:
- **Exact matches**: Auto-verified, low priority
- **Substring matches**: Need review to determine if valid alternative
- **No matches**: Need review to determine if gold is wrong or model failed

---

# Multi-Model Workflow (Recommended)

For more robust platinum benchmarks, use multiple models to identify consensus and disagreements.

## Step 1: Multi-Model Evaluation

Run multiple models in a single command:

```bash
# Evaluate with 3 models
python create_platinum_multi.py \
  --dataset 2wikimultihopqa \
  --models gpt-4o gpt-4o-mini claude-3-5-sonnet-20241022 \
  --num-questions 50

# Or with more models
python create_platinum_multi.py \
  --dataset musique \
  --models gpt-4o gpt-4o-mini claude-3-5-sonnet-20241022 gemini-2.0-flash-exp \
  --num-questions 100
```

### Output Structure

Creates a directory with all results:

```
platinum_review/2wikimultihopqa_20251021_163045/
├── metadata.json                          # Overall stats
├── gpt-4o.json                           # Individual model results
├── gpt-4o-mini.json                      # Individual model results
├── claude-3-5-sonnet-20241022.json       # Individual model results
└── combined_review.json                  # MAIN REVIEW FILE ⭐
```

### What Happens

- Runs each model sequentially on all questions
- Saves individual model results to separate files
- Aggregates into `combined_review.json` with:
  - All model predictions per question
  - Match types per model (exact/substring/none)
  - Agreement metrics
  - **Auto-verify**: Only if ALL models have exact match

### Auto-Verification Logic

```
if ALL models have exact match:
    review_status = "verified"
    platinum_target = gold_answers
else:
    review_status = "pending"
    platinum_target = null
```

## Step 2: Review Multi-Model Results

Launch the UI and select the directory:

```bash
streamlit run review_ui.py
```

In the UI:
1. Select the directory (📁 icon) from the dropdown
2. UI automatically loads `combined_review.json`
3. See all model predictions in a table for each question

### Multi-Model UI Features

- **Model predictions table**: See all model answers side-by-side
- **Agreement indicators**:
  - ✅ All exact match (auto-verified)
  - ⚠️ Some exact match (needs review)
  - ❌ No exact match (high priority)
- **Filter by agreement**: Focus on disagreements
- **Match type per model**: See which models got exact/substring/none

## Step 3: Review Strategy

### Priority Order

1. **High Priority**: No exact matches (all models wrong or question is bad)
2. **Medium Priority**: Some exact matches (identify valid alternatives)
3. **Low Priority**: All exact matches (already auto-verified)

### Example: Multi-Model Review

```
Question: "How many members are in the Florida Senate?"
Gold: ["40 members"]

Model Predictions:
  gpt-4o: "40 members" (exact match ✅)
  gpt-4o-mini: "40" (substring match ⚠️)
  claude-3-5-sonnet: "forty" (no match ❌)

Agreement: Some exact match

Decision: REVISED
Reason: All models are semantically correct, just different formats
Platinum target: ["40 members", "40", "forty", "forty members"]
```

### When Models Disagree

If models give different answers:

1. **Check the oracle context** - Is the answer clearly stated?
2. **Answer it yourself** - What's the correct answer from context?
3. **Evaluate each model's answer**:
   - Are some models correct and others wrong? → `verified`
   - Are different phrasings all correct? → `revised`
   - Is the question ambiguous/unanswerable? → `rejected`

## Step 4: Save and Export

1. Click **"💾 Save Changes"** in the UI
2. Changes are saved to `combined_review.json`
3. Individual model files remain unchanged (for reference)

---

## Single vs Multi-Model: When to Use Each

### Single-Model (`create_platinum_simple.py`)

**Use when:**
- Quick prototyping with 10-50 questions
- Limited API budget
- Testing the workflow
- Only have access to one model

**Pros:**
- Faster execution
- Lower cost
- Simpler review

**Cons:**
- Less robust verification
- May miss valid alternative answers
- No consensus-based validation

### Multi-Model (`create_platinum_multi.py`)

**Use when:**
- Creating production platinum benchmarks
- Have 100+ questions to review
- Want high-quality, consensus-based verification
- Need to identify valid alternative answer formats

**Pros:**
- Consensus-based auto-verification (all models must agree)
- Identifies valid alternative answers
- More robust quality control
- Highlights problematic questions (when models disagree)

**Cons:**
- Higher API costs
- Slower execution
- More complex review (but better quality)

---

## Tips

- **Start small**: Test with 10-20 questions first to calibrate
- **Use multi-model for final**: Single-model for prototyping, multi-model for production
- **Focus on disagreements**: When models disagree, question quality is often the issue
- **Document decisions**: Use review notes field for tricky cases
- **Be consistent**: Establish rules for answer formatting early
- **Iterate**: Review in batches, adjust criteria as you learn patterns
