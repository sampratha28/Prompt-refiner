task_requirement = '''i have customer reviews as input. i want to put them under one of the following categories based on their sentiment - Neutral/Positive/Negative.' \
Also, i need you to categorise the review into one of the following -  Refund/Delivery/Pricing/Customer Support/Product Quality/Account'''

guidelines = ['output should be always one of the following. never conflicting. if multiple matches arise, pick the most suitable one.',
"output should be json format with keys 'sentiment'and 'topic' and values being one of Neutral/Positive/Negative and Refund/Delivery/Pricing/Customer Support/Product Quality respectively.",
'there should be no pretext or post text. Only the desired json output.']

output_strut = {"sentiment": "<either Neutral/Positive/Negative>",
                "topic": "<either Refund/Delivery/Pricing/Customer Support/Product Quality/Account>"
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
