An end to end prompt optimization framework that automatically improves LLM prompts using an iterative feedback loop:

Prompt Generation → Inference → Evaluation → Feedback → Refinement

✨ Overview

This project helps you systematically improve prompt quality by:

Generating structured prompts using LLMs
Running them on real datasets
Evaluating outputs with an agentic evaluator
Iteratively refining prompts based on failure patterns
🧠 Key Capabilities
🔁 Automated Prompt Refinement Loop
🤖 Agentic Evaluation System (plan → execute → report)
📊 Granular Metrics Tracking
🎯 Benchmark-driven Optimization
⛔ Early Stopping with Patience
🧪 Dataset-based Testing
🏗️ Architecture
          ┌────────────────────┐
          │  Prompt Generator  │
          └────────┬───────────┘
                   │
                   ▼
          ┌────────────────────┐
          │     Inference      │
          └────────┬───────────┘
                   │
                   ▼
          ┌────────────────────┐
          │   Agent Evaluator  │
          └────────┬───────────┘
                   │
                   ▼
          ┌────────────────────┐
          │ Feedback Generator │
          └────────┬───────────┘
                   │
                   ▼
          🔁 Iteration Loop
📂 Project Structure
.
├── main.py
├── setting.py
├── helper.py
├── best_prompt.json
├── final_metrics.json
├── iteration_history.json
├── optimization_progress.csv
└── README.md
⚙️ Installation
git clone https://github.com/your-username/prompt-optimization-pipeline.git
cd prompt-optimization-pipeline
pip install pandas openpyxl
🛠️ Configuration

Update setting.py:

TARGET_ACCURACY = 0.90
TARGET_AVG_SCORE = 85
MAX_ITERATIONS = 10
PATIENCE = 3

Also configure:

PROMPT_WRITER_MODEL
INFERENCE_MODEL
EVALUATOR_MODEL
▶️ Usage
python main.py
📥 Input Dataset

Excel file: customer_support_eval_dataset.xlsx

Required columns:

Column Name	Description
Review Text	Input text
Ground Truth Sentiment	Label
Ground Truth Topic	Label
📤 Outputs
File	Description
best_prompt.json	Final optimized prompt
final_metrics.json	Final evaluation metrics
iteration_history.json	Iteration-wise results
optimization_progress.csv	Progress tracking
📊 Evaluation Metrics
Metric	Weight	Description
Format Compliance	20%	JSON/schema correctness
Completeness	20%	Required fields present
Correctness	60%	Match with ground truth
Overall Score = 0.2 * Format + 0.2 * Completeness + 0.6 * Correctness
🔁 Optimization Loop
1. Generate Prompt
2. Run Inference
3. Evaluate Outputs
4. Identify Failures
5. Generate Feedback
6. Refine Prompt
7. Repeat until benchmark met
🧪 Example Iteration
Iteration 1 → Format errors ❌  
Iteration 2 → Better structure ⚠️  
Iteration 3 → High accuracy ✅ (Stop)
🧩 Core Components
Function	Purpose
optimize_prompt	Main orchestration loop
prompt_writer	Prompt generation/refinement
perform_inference	Runs model predictions
run_evaluation	Agentic evaluation
generate_refinement_feedback	Feedback generation
⚠️ Notes
Requires LLM API integration (helper.call_llm)
Evaluation is LLM-based (agentic), not purely deterministic
Start with small datasets for faster iteration
🔮 Future Improvements
⚡ Parallel inference
🧪 Deterministic evaluation tools
📊 Dashboard for experiment tracking
🔌 Integration with MLflow / Weights & Biases
