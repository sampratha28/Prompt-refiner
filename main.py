import pandas as pd
import setting
import helper
import json
from typing import Dict, List, Any, Tuple, Optional

def prompt_writer(requirement: str, guidelines: str, output_format: str, feedback: Optional[str] = None, model_name: str = setting.default_model) -> str:
    """
    Generate or refine a prompt based on task requirements and optional feedback.
    
    Args:
        requirement: Task description
        guidelines: Evaluation criteria
        output_format: Desired output structure
        feedback: Optional feedback from previous iteration (for refinement)
        model_name: Model to use for prompt generation
    
    Returns:
        Generated prompt as JSON string
    """
    
    system = """You are an expert prompt engineer specializing in creating precise, effective prompts for AI models.

Your task is to analyze the given task description, guidelines/criteria, and desired output format, then construct a complete prompt pair consisting of:

1. **system prompt** - Sets the AI's role, behavior, tone, and core instructions
2. **user prompt** - Delivers the specific task with a clear $input placeholder for dynamic content

**Prompt Engineering Standards to Follow:**
- Be specific and explicit about role, task, and constraints
- Use clear section headers within prompts when needed
- Include examples only when explicitly requested
- Define success criteria and evaluation standards
- Handle edge cases and error conditions
- Maintain consistent tone matching the task requirements
- Use structured formatting (bullets, numbered lists) for clarity
- Ensure $input placeholder is used exactly once in user prompt
- For classification tasks: explicitly define all possible labels and decision criteria
- For structured output: enforce exact JSON schema with required fields

**IF FEEDBACK IS PROVIDED:**
- Analyze failure patterns from previous iteration
- Address specific issues mentioned (format errors, misclassifications, missing fields, etc.)
- Add explicit instructions to prevent recurring mistakes
- Strengthen weak areas while maintaining clarity

**Output Format (JSON only, no additional text):**
```json
{
  "system": "Complete system prompt text here",
  "user": "Complete user prompt with $input placeholder here"
}
```"""

    if feedback:
        user = f"""Task description: {requirement}
Guidelines/criterias: {guidelines}
Desired output format: {output_format}

PREVIOUS ITERATION FEEDBACK (CRITICAL - ADDRESS THESE ISSUES):
{feedback}

Refine the prompt to specifically fix the identified failures while maintaining all original requirements.
Focus on:
1. Adding explicit instructions to prevent format violations
2. Clarifying ambiguous decision criteria
3. Strengthening edge case handling
4. Improving label definitions for classification tasks"""
    else:
        user = f"""Task description: {requirement}
Guidelines/criterias: {guidelines}
Desired output format: {output_format}

Create an optimal prompt from scratch."""
    
    prompt = helper.call_llm(model_name, system, user)
    return prompt


# =============================================================================
# INFERENCE
# =============================================================================

def perform_inference(messages: Dict[str, str], data: pd.DataFrame, input_column: str = "Review Text", model: str = setting.default_model) -> Tuple[List[str], List[Any]]:
    """
    Run inference on dataset using generated prompt.
    
    Args:
        messages: Dict with 'system' and 'user' prompt
        data: DataFrame with input data
        input_column: Column name containing input text
        model: Model to use for inference
    
    Returns:
        Tuple of (raw_results, postprocessed_results)
    """
    texts = data[input_column].to_list()
    raw = []
    postprocessed = []
    
    print(f"Running inference on {len(texts)} samples...")
    
    for idx, input_text in enumerate(texts):
        user_prompt = messages['user'].replace('$input', input_text)
        result = helper.call_llm(model, messages['system'], user_prompt)
        postprocessed_res = helper.extract_json_from_text(result)
        
        raw.append(result)
        postprocessed.append(postprocessed_res)
        
        if (idx + 1) % 10 == 0:
            print(f"  Progress: {idx + 1}/{len(texts)}")
    
    return raw, postprocessed


# =============================================================================
# METRICS
# =============================================================================

def compute_aggregate_metrics(per_row_evals: List[Dict]) -> Dict[str, Any]:
    """
    Compute aggregate metrics from per-row evaluations.
    """
    total = len(per_row_evals)
    passed = 0
    partial = 0
    failed = 0
    
    scores = []
    format_scores = []
    completeness_scores = []
    correctness_scores = []
    
    for eval_record in per_row_evals:
        evaluation = eval_record.get("evaluation", {})
        report = evaluation.get("report", {})
        
        overall_result = report.get("overall_result", "fail")
        if overall_result == "pass":
            passed += 1
        elif overall_result == "partial":
            partial += 1
        else:
            failed += 1
        
        score = report.get("score", 0)
        scores.append(score)
        
        score_breakdown = report.get("score_breakdown", {})
        format_scores.append(score_breakdown.get("format_compliance", 0))
        completeness_scores.append(score_breakdown.get("completeness", 0))
        correctness_scores.append(score_breakdown.get("correctness", 0))
    
    def safe_avg(lst):
        return sum(lst) / len(lst) if lst else 0.0
    
    return {
        "total_samples": total,
        "passed": passed,
        "partial": partial,
        "failed": failed,
        "pass_rate": passed / total if total > 0 else 0.0,
        "average_score": safe_avg(scores),
        "average_format_compliance": safe_avg(format_scores),
        "average_completeness": safe_avg(completeness_scores),
        "average_correctness": safe_avg(correctness_scores),
        "score_distribution": {
            "pass": passed,
            "partial": partial,
            "fail": failed
        }
    }

def run_evaluation(raw_results: List[str], ground_truths: List[Any], inputs: List[str], task_requirement: str, guidelines: str, output_struct: str, model: str = setting.default_model) -> Dict[str, Any]:
    """
    Evaluate candidate results against ground truth using an agentic workflow.
    """
    
    system_prompt = """
You are an autonomous evaluation agent with planning, tool-selection, execution, replanning, and reporting capabilities.

Your purpose is to evaluate a candidate result against:
- a task description
- evaluation criteria
- expected output format
- GROUND TRUTH (the correct answer)

You must act like an execution-capable agent that can reason about what to verify, decide which tools are necessary, run checks, and produce an evidence-based report.

You are not a passive reviewer. You must follow the full agent loop:
UNDERSTAND -> PLAN -> SELECT TOOLS -> EXECUTE -> OBSERVE -> REPLAN IF NEEDED -> REPORT

==================================================
AGENT ROLE
==================================================

You are responsible for:
- understanding the task objective and constraints
- identifying explicit and implicit success criteria
- decomposing evaluation into verifiable checks
- comparing candidate output against GROUND TRUTH
- deciding whether tool usage is necessary
- executing deterministic validations where possible
- validating structure, logic, completeness, correctness, and compliance
- generating a final evidence-based evaluation report

==================================================
AVAILABLE TOOLS
==================================================

You may use the following abstract tool types. Treat them as callable capabilities.

1. search
Purpose: verify factual claims, check latest external information
Use when: candidate or ground truth includes factual claims requiring external grounding

2. fetch
Purpose: retrieve full content from a URL or document
Use when: source documents must be inspected directly

3. execute_code
Purpose: run code for schema validation, parsing, calculations, unit tests, consistency checks, scoring logic
Use when: JSON/schema/code/math/table/output needs programmatic verification

4. read_file
Purpose: inspect provided files, logs, reports, datasets
Use when: evaluation requires file inspection

5. no_tool
Purpose: complete evaluation without external calls
Use when: all checks can be done from provided text alone

==================================================
TOOL DECISION POLICY
==================================================

Before using any tool, decide:
- what exactly needs verification
- why the tool is necessary
- what evidence the tool is expected to provide
- whether the check can be done without the tool

Rules:
- Do not use tools performatively
- Prefer deterministic checks over subjective checks
- Use execute_code for structured validation whenever possible
- Use search/fetch only when factual verification genuinely requires external evidence
- If a planned tool call fails or yields insufficient evidence, replan
- If no tool is needed, explicitly state why

==================================================
MANDATORY AGENTIC WORKFLOW
==================================================

Follow this exact sequence.

### Step 1: Understand
Extract and normalize:
- task objective
- expected behavior
- constraints
- desired output format
- evaluation criteria
- hidden or implied requirements

### Step 2: Plan
Create a concrete plan before any execution.

Your plan must include:
- checks to perform (format, completeness, correctness vs ground truth, edge cases)
- test cases to run
- tools required for each check
- scoring logic (how to compare candidate vs ground truth)
- stop conditions

### Step 3: Select Tools
For each planned check, specify:
- tool name
- why it is needed
- input to the tool
- expected evidence
- fallback if the tool fails

### Step 4: Execute
Run the checks in this order:
1. output format correctness
2. required fields and completeness
3. instruction compliance
4. internal consistency
5. CORRECTNESS VS GROUND TRUTH (most important)
6. factual correctness (if no ground truth available)
7. quality and edge-case handling

### Step 5: Observe
For each executed step, record:
- what was checked
- result: pass/fail/partial/not_executed
- evidence (quote candidate and ground truth where relevant)
- confidence level: high/medium/low
- whether replanning is needed

### Step 6: Replan if Needed
Trigger replanning when:
- a tool fails
- evidence is incomplete
- a new issue is discovered
- a prior assumption is invalidated

### Step 7: Report
Generate a structured final report with evidence.

Do not jump directly to the verdict.

==================================================
SPECIALIZED EVALUATION RULES
==================================================

If the task involves JSON or structured output:
- verify valid parseability
- verify required keys match ground truth schema
- verify value types
- compare candidate values against ground truth values
- compute field-level accuracy

If the task involves classification:
- compare predicted label vs ground truth label
- verify explanation quality
- check consistency between label and rationale
- mark correct/incorrect with evidence

If the task involves extraction:
- verify all required fields are extracted
- compare extracted values against ground truth
- check for hallucinated fields
- compute extraction accuracy

If the task involves summarization/generation:
- check completeness vs ground truth
- check for hallucinations (claims not in ground truth or input)
- check instruction following
- ignore stylistic differences unless style is part of criteria

==================================================
SCORING LOGIC
==================================================

Use this scoring approach:
- format_compliance: 0-100 (based on schema/format adherence)
- completeness: 0-100 (based on required fields present)
- correctness: 0-100 (based on match with ground truth)
- overall_score: weighted average (format: 20%, completeness: 20%, correctness: 60%)

overall_result:
- "pass" if overall_score >= 80
- "partial" if 50 <= overall_score < 80
- "fail" if overall_score < 50

==================================================
OUTPUT FORMAT
==================================================

Return valid JSON only. No markdown, no conversational filler.

{
  "understanding": {
    "objective": "...",
    "constraints": ["...", "..."],
    "expected_output": "...",
    "success_criteria": ["...", "..."]
  },
  "plan": {
    "checks": [
      {
        "name": "...",
        "purpose": "...",
        "method": "...",
        "tool": "search|fetch|execute_code|read_file|no_tool",
        "reason_for_tool": "...",
        "expected_evidence": "...",
        "fallback": "..."
      }
    ],
    "test_cases": [
      {
        "name": "...",
        "input_or_condition": "...",
        "expected_behavior": "..."
      }
    ],
    "scoring_method": {
      "type": "weighted",
      "weights": {
        "format_compliance": 0.2,
        "completeness": 0.2,
        "correctness": 0.6
      },
      "details": "..."
    },
    "termination_condition": "..."
  },
  "execution": {
    "steps": [
      {
        "check_name": "...",
        "tool_used": "...",
        "status": "pass|fail|partial|not_executed",
        "evidence": {
          "candidate": "...",
          "ground_truth": "...",
          "comparison": "..."
        },
        "confidence": "high|medium|low",
        "needs_replan": false
      }
    ],
    "replanning": []
  },
  "report": {
    "overall_result": "pass|fail|partial",
    "score": 0,
    "score_breakdown": {
      "format_compliance": 0,
      "completeness": 0,
      "correctness": 0
    },
    "summary": "...",
    "strengths": ["...", "..."],
    "issues": ["...", "..."],
    "recommendations": ["...", "..."],
    "unverified_items": []
  }
}

==================================================
BEHAVIORAL RULES
==================================================

- Be objective and evidence-based
- Always compare candidate against ground truth when available
- Do not claim execution if no execution occurred
- Do not invent tool results
- If a claim cannot be verified, mark it unverified
- If information is insufficient, say so explicitly
- Do not output conversational filler
- Do not output markdown
- Output must be valid JSON only
"""

    per_row_evals = []
    
    for idx, (candidate, ground_truth, input_text) in enumerate(zip(raw_results, ground_truths, inputs)):
        print(f"  Evaluating sample {idx + 1}/{len(raw_results)}")
        
        user_prompt = f"""
Task Description:
{task_requirement}

Evaluation Criteria:
{guidelines}

Expected Output Format:
{output_struct}

Input:
{input_text}

Ground Truth (Correct Answer):
{json.dumps(ground_truth, indent=2) if isinstance(ground_truth, (dict, list)) else ground_truth}

Candidate Result (Model Output):
{candidate}

Evaluate the candidate result against the ground truth using the full agentic workflow.
Compare candidate values directly against ground truth values.
Identify mismatches, missing fields, hallucinations, and format violations.
Compute field-level and overall accuracy.
Return valid JSON only.
"""
        
        eval_response = helper.call_llm(model, system_prompt, user_prompt)
        try:
            eval_json = helper.extract_json_from_text(eval_response)
            if isinstance(eval_json, list) and len(eval_json) > 0:
                eval_json = eval_json[0]
        except Exception as e:
            print(f"Warning: Could not parse eval JSON for sample {idx + 1}: {e}")
            eval_json = {"error": "Failed to parse evaluation response", "raw": eval_response}
        
        eval_record = {
            "sample_id": idx,
            "input": input_text,
            "candidate": candidate,
            "ground_truth": ground_truth,
            "evaluation": eval_json
        }
        
        per_row_evals.append(eval_record)
    aggregate_metrics = compute_aggregate_metrics(per_row_evals)
    return {
        "per_row_evals": per_row_evals,
        "aggregate_metrics": aggregate_metrics,
        "total_samples": len(raw_results)
    }

def generate_refinement_feedback(eval_result: Dict[str, Any]) -> str:
    """
    Analyze evaluation results and generate actionable feedback for prompt refinement.
    """
    metrics = eval_result["aggregate_metrics"]
    per_row = eval_result["per_row_evals"]
    
    feedback_parts = []
    
    # Overall performance summary
    feedback_parts.append(f"OVERALL PERFORMANCE:")
    feedback_parts.append(f"- Pass Rate: {metrics['pass_rate']:.2%} (target: {setting.TARGET_ACCURACY:.2%})")
    feedback_parts.append(f"- Average Score: {metrics['average_score']:.2f} (target: {setting.TARGET_AVG_SCORE})")
    feedback_parts.append(f"- Format Compliance: {metrics['average_format_compliance']:.2f}")
    feedback_parts.append(f"- Completeness: {metrics['average_completeness']:.2f}")
    feedback_parts.append(f"- Correctness: {metrics['average_correctness']:.2f}")
    feedback_parts.append("")
    
    # Identify failure patterns
    feedback_parts.append("FAILURE PATTERNS IDENTIFIED:")
    
    format_failures = []
    correctness_failures = []
    completeness_failures = []
    
    for record in per_row:
        evaluation = record.get("evaluation", {})
        report = evaluation.get("report", {})
        score_breakdown = report.get("score_breakdown", {})
        issues = report.get("issues", [])
        
        if score_breakdown.get("format_compliance", 100) < 70:
            format_failures.append({
                "sample_id": record["sample_id"],
                "input": record["input"][:100],
                "issues": issues
            })
        
        if score_breakdown.get("correctness", 100) < 70:
            correctness_failures.append({
                "sample_id": record["sample_id"],
                "input": record["input"][:100],
                "issues": issues
            })
        
        if score_breakdown.get("completeness", 100) < 70:
            completeness_failures.append({
                "sample_id": record["sample_id"],
                "input": record["input"][:100],
                "issues": issues
            })
    
    if format_failures:
        feedback_parts.append(f"\n1. FORMAT FAILURES ({len(format_failures)} samples):")
        for f in format_failures[:3]:  # Show top 3
            feedback_parts.append(f"   - Sample {f['sample_id']}: {', '.join(f['issues'][:2])}")
        feedback_parts.append("   => Add stricter JSON schema enforcement and format validation instructions")
    
    if correctness_failures:
        feedback_parts.append(f"\n2. CORRECTNESS FAILURES ({len(correctness_failures)} samples):")
        for f in correctness_failures[:3]:
            feedback_parts.append(f"   - Sample {f['sample_id']}: {', '.join(f['issues'][:2])}")
        feedback_parts.append("   => Clarify decision criteria, add explicit label definitions, provide edge case guidance")
    
    if completeness_failures:
        feedback_parts.append(f"\n3. COMPLETENESS FAILURES ({len(completeness_failures)} samples):")
        for f in completeness_failures[:3]:
            feedback_parts.append(f"   - Sample {f['sample_id']}: {', '.join(f['issues'][:2])}")
        feedback_parts.append("   => Add explicit checklist of required fields, emphasize completeness requirement")
    
    feedback_parts.append("")
    feedback_parts.append("PRIORITY ACTIONS FOR NEXT ITERATION:")
    if metrics['average_format_compliance'] < 80:
        feedback_parts.append("1. CRITICAL: Fix format violations - add JSON schema template and validation rules")
    if metrics['average_correctness'] < 80:
        feedback_parts.append("2. CRITICAL: Improve accuracy - clarify classification criteria and decision boundaries")
    if metrics['average_completeness'] < 80:
        feedback_parts.append("3. IMPORTANT: Ensure all required fields are present - add mandatory field checklist")
    
    return "\n".join(feedback_parts)

def check_benchmark(metrics: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Check if evaluation metrics meet benchmark thresholds.
    
    Returns:
        Tuple of (passed, reason)
    """
    reasons = []
    
    if metrics['pass_rate'] < setting.TARGET_ACCURACY:
        reasons.append(f"Pass rate {metrics['pass_rate']:.2%} < target {setting.TARGET_ACCURACY:.2%}")
    
    if metrics['average_score'] < setting.TARGET_AVG_SCORE:
        reasons.append(f"Average score {metrics['average_score']:.2f} < target {setting.TARGET_AVG_SCORE}")
    
    format_failure_rate = 1 - (metrics['average_format_compliance'] / 100)
    if format_failure_rate > setting.MAX_FORMAT_FAILURES:
        reasons.append(f"Format failure rate {format_failure_rate:.2%} > max {setting.MAX_FORMAT_FAILURES:.2%}")
    
    passed = len(reasons) == 0
    reason = "All benchmarks met" if passed else "; ".join(reasons)
    
    return passed, reason


# =============================================================================
# MAIN OPTIMIZATION LOOP
# =============================================================================

def optimize_prompt(df: pd.DataFrame, task_requirement: str, guidelines: List[str], output_struct: str, input_column: str , res_columns: List[str]) -> Dict[str, Any]:
    """
    Main optimization loop: generate prompt -> infer -> evaluate -> refine until benchmark met.
    
    Args:
        df: DataFrame with test data
        task_requirement: Task description
        guidelines: Evaluation criteria
        output_struct: Expected output format
        input_column: Column with input text
        sentiment_column: Column with ground truth sentiment
        topic_column: Column with ground truth topic
    
    Returns:
        Final results including best prompt, metrics, and iteration history
    """
    
    print("="*80)
    print("PROMPT OPTIMIZATION PIPELINE")
    print("="*80)
    print(f"Target Accuracy: {setting.TARGET_ACCURACY:.2%}")
    print(f"Target Avg Score: {setting.TARGET_AVG_SCORE}")
    print(f"Max Iterations: {setting.MAX_ITERATIONS}")
    print(f"Patience level: {setting.PATIENCE}")
    print("="*80)
    patience = 1
    ground_truths = helper.prepare_ground_truths(df, res_columns)
    
    inputs = df[input_column].to_list()
    
    iteration_history = []
    best_prompt = None
    best_metrics = None
    best_iteration = -1
    
    current_feedback = None
    
    for iteration in range(1, setting.MAX_ITERATIONS + 1):
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration}/{setting.MAX_ITERATIONS}")
        print(f"PATIENCE {patience}/{setting.PATIENCE}")
        print(f"{'='*80}")
        print("\nGenerating prompt...")
        if current_feedback:
            print(f"Using feedback from previous iteration to refine prompt")
        prompt_str = prompt_writer(task_requirement, guidelines, output_struct, current_feedback, setting.PROMPT_WRITER_MODEL)
        prompt_messages = helper.extract_json_from_text(prompt_str)
        if isinstance(prompt_messages, list) and len(prompt_messages) > 0:
            prompt_messages = prompt_messages[0]
        
        print(f"Prompt generated successfully\n\n",prompt_messages)
        print("\nRunning inference...")
        raw_results, fin_results = perform_inference(
            prompt_messages,
            df,
            input_column=input_column,
            model=setting.INFERENCE_MODEL
        )
        print("\nEvaluating results...")
        eval_result = run_evaluation(raw_results, ground_truths, inputs, task_requirement, guidelines, output_struct, setting.EVALUATOR_MODEL)
        metrics = eval_result["aggregate_metrics"]
        print(f"\n{iteration} Results:")
        print(f"  Pass Rate: {metrics['pass_rate']:.2%}")
        print(f"  Average Score: {metrics['average_score']:.2f}")
        print(f"  Format Compliance: {metrics['average_format_compliance']:.2f}")
        print(f"  Completeness: {metrics['average_completeness']:.2f}")
        print(f"  Correctness: {metrics['average_correctness']:.2f}")
        iteration_record = {
            "iteration": iteration,
            "prompt": prompt_messages,
            "metrics": metrics,
            "eval_result": eval_result
        }
        iteration_history.append(iteration_record)
        
        benchmark_passed, reason = check_benchmark(metrics)
        
        if benchmark_passed:
            print(f"\n✓ BENCHMARK MET! Stopping optimization.")
            print(f"  Reason: {reason}")
            best_prompt = prompt_messages
            best_metrics = metrics
            best_iteration = iteration
            break
        else:
            print(f"\n✗ Benchmark not met: {reason}")
            if best_metrics is None or metrics['pass_rate'] > best_metrics['pass_rate']:
                best_prompt = prompt_messages
                best_metrics = metrics
                best_iteration = iteration
                print(f"  -> New best iteration (pass rate: {metrics['pass_rate']:.2%})")
            if iteration > 1:
                prev_metrics = iteration_history[-2]["metrics"]
                improvement = metrics['pass_rate'] - prev_metrics['pass_rate']
                if improvement < setting.MIN_IMPROVEMENT:
                    patience += 1
                else:
                    patience = 0
                print(f"\n⚠ Improvement below threshold ({improvement:.2%} < {setting.MIN_IMPROVEMENT:.2%})")
                print(f"  Consider stopping early or adjusting prompt strategy")
            
            if patience >= setting.PATIENCE:
                print("Early stopping due to patience limit")
                break

            if iteration < setting.MAX_ITERATIONS:
               
                print(f"\nGenerating refinement feedback...")
                current_feedback = generate_refinement_feedback(eval_result)
                print("\n" + "-"*80)
                print("FEEDBACK FOR NEXT ITERATION:")
                print("-"*80)
                print(current_feedback)
                print("-"*80)
    
    # Final summary
    print(f"\n{'='*80}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"Best Iteration: {best_iteration}")
    print(f"Best Pass Rate: {best_metrics['pass_rate']:.2%}")
    print(f"Best Average Score: {best_metrics['average_score']:.2f}")
    print(f"Total Iterations: {len(iteration_history)}")
    print(f"{'='*80}")
    
    return {
        "best_prompt": best_prompt,
        "best_metrics": best_metrics,
        "best_iteration": best_iteration,
        "iteration_history": iteration_history,
        "benchmark_met": check_benchmark(best_metrics)[0],
        "total_iterations": len(iteration_history)
    }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    print("Loading dataset...")
    df = pd.read_excel('customer_support_eval_dataset.xlsx')
    print(f"Dataset loaded: {len(df)} samples") 
    final_result = optimize_prompt(df, setting.task_requirement, setting.guidelines, setting.output_strut, "Review Text", ["Ground Truth Sentiment", "Ground Truth Topic"])
    
    # Save results
    print("\nSaving results...")
    
    # Save best prompt
    with open("best_prompt.json", "w") as f:
        json.dump(final_result["best_prompt"], f, indent=2)
    
    # Save metrics
    with open("final_metrics.json", "w") as f:
        json.dump(final_result["best_metrics"], f, indent=2)
    
    # Save detailed iteration history
    iteration_summary = []
    for iter_record in final_result["iteration_history"]:
        iteration_summary.append({
            "iteration": iter_record["iteration"],
            "metrics": iter_record["metrics"]
        })
    
    with open("iteration_history.json", "w") as f:
        json.dump(iteration_summary, f, indent=2)
    
    # Save CSV summary
    summary_df = pd.DataFrame([
        {
            "iteration": r["iteration"],
            "pass_rate": r["metrics"]["pass_rate"],
            "average_score": r["metrics"]["average_score"],
            "format_compliance": r["metrics"]["average_format_compliance"],
            "completeness": r["metrics"]["average_completeness"],
            "correctness": r["metrics"]["average_correctness"],
            "passed": r["metrics"]["pass_rate"] >= setting.TARGET_ACCURACY
        }
        for r in iteration_summary
    ])
    summary_df.to_csv("optimization_progress.csv", index=False)
    
    print("\nResults saved:")
    print("  - best_prompt.json")
    print("  - final_metrics.json")
    print("  - iteration_history.json")
    print("  - optimization_progress.csv")
    
    # Print final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Benchmark Met: {final_result['benchmark_met']}")
    print(f"Best Pass Rate: {final_result['best_metrics']['pass_rate']:.2%}")
    print(f"Best Average Score: {final_result['best_metrics']['average_score']:.2f}")
    print(f"Iterations Used: {final_result['total_iterations']}/{setting.MAX_ITERATIONS}")
    print("="*80)
