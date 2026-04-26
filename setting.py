# task_requirement = '''i have customer reviews as input. i want to put them under one of the following categories based on their sentiment - Neutral/Positive/Negative.' \
# Also, i need you to categorise the review into one of the following -  Refund/Delivery/Pricing/Customer Support/Product Quality/Account'''

# guidelines = ['output should be always one of the following. never conflicting. if multiple matches arise, pick the most suitable one.',
# "output should be json format with keys 'sentiment'and 'topic' and values being one of Neutral/Positive/Negative and Refund/Delivery/Pricing/Customer Support/Product Quality respectively.",
# 'there should be no pretext or post text. Only the desired json output.']

# output_strut = {"sentiment": "<either Neutral/Positive/Negative>",
#                 "topic": "<either Refund/Delivery/Pricing/Customer Support/Product Quality/Account>"
#                 }

task_requirement = """I have pairs of compliance requirements (text1) and implementation evidence (text2).
I want to assess how well the implementation covers the requirement and classify the coverage into one of the following:
No Match / Minimal / Substantial / Strong."""

guidelines = [
    "Output should always be exactly one of the four coverage labels: No Match, Minimal, Substantial, Strong. Never output conflicting or multiple labels.",
    "No Match: The implementation has no relevance to the requirement — no controls, policies, or procedures address the obligation.",
    "Minimal: The implementation acknowledges the requirement but lacks actionable controls, is outdated, unapproved, or only informally addressed.",
    "Substantial: The implementation addresses the core requirement with documented controls but has minor gaps such as incomplete rollout, missing monitoring, or pending reviews.",
    "Strong: The implementation fully satisfies the requirement with documented policy, designated ownership, evidence of operation, monitoring, and audit or certification.",
    "Output must be JSON format with keys 'coverage' and 'explanation'. 'coverage' must be exactly one of: No Match, Minimal, Substantial, Strong.",
    "The 'explanation' must be 1-2 sentences justifying the label with direct reference to what is present or absent in the implementation.",
    "There should be no pretext or post text. Only the desired JSON output."
]

output_strut = {
    "coverage": "<one of: No Match / Minimal / Substantial / Strong>",
    "explanation": "<1-2 sentence justification referencing the implementation evidence>"
}

TARGET_ACCURACY = 0.85          # 85% pass rate required
TARGET_AVG_SCORE = 80.0         # Average score >= 80
MAX_FORMAT_FAILURES = 0.10      # Max 10% format failures allowed

MAX_ITERATIONS = 5              # Max prompt refinement attempts
MIN_IMPROVEMENT = 0.05          # Minimum 5% improvement to continue

PATIENCE = 3

PROMPT_WRITER_MODEL = "llama-3.3-70b-versatile"
INFERENCE_MODEL = "llama-3.3-70b-versatile"
EVALUATOR_MODEL = "llama-3.3-70b-versatile"

default_model = "llama-3.3-70b-versatile"


ground_truth_columns = ["Coverage", "Explanation"]
input_columns = ["Requirement (text1)","Implementation / Evidence (text2)"]