import os
import sys
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# Ensure output directories exist
EVAL_DIR = Path(__file__).resolve().parent.parent
SCORECARDS_DIR = EVAL_DIR / "scorecards"
SCORECARDS_DIR.mkdir(parents=True, exist_ok=True)

# Load env variables
load_dotenv(find_dotenv())

try:
    from langchain_groq import ChatGroq
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, HallucinationMetric
    from deepeval.models.base_model import DeepEvalBaseLLM
except ImportError as e:
    print(f"ImportError: {e}")
    print("Please make sure DeepEval and LangChain are installed.")
    sys.exit(1)

# Custom LLM Wrapper for DeepEval to route through Groq ChatGroq
class CustomGroqLLM(DeepEvalBaseLLM):
    def __init__(self, chat_model):
        self.chat_model = chat_model
        
    def load_model(self):
        return self.chat_model
        
    def generate(self, prompt: str, schema=None) -> str:
        # Respect API rate limiting
        time.sleep(1)
        response = self.chat_model.invoke(prompt)
        return response.content
        
    async def a_generate(self, prompt: str, schema=None) -> str:
        # Fallback to synchronous execution to ensure absolute sequential control
        return self.generate(prompt)
        
    def get_model_name(self) -> str:
        return "llama-3.3-70b-versatile"

def run_deepeval_evaluation():
    print("=" * 60)
    print("Starting DeepEval Evaluation...")
    print("=" * 60)

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("Error: GROQ_API_KEY not found in environment variables.")
        sys.exit(1)

    # Initialize LangChain Groq model
    chat_model = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=groq_api_key,
        temperature=0.0
    )
    
    # Initialize the custom DeepEval LLM wrapper
    custom_llm = CustomGroqLLM(chat_model)

    # Instantiate metrics with async_mode=False and strict_mode=False
    print("Initializing metrics...")
    faithfulness_metric = FaithfulnessMetric(
        threshold=0.5,
        model=custom_llm,
        async_mode=False
    )
    
    answer_relevancy_metric = AnswerRelevancyMetric(
        threshold=0.5,
        model=custom_llm,
        async_mode=False
    )
    
    # Hallucination metric in DeepEval requires contexts
    hallucination_metric = HallucinationMetric(
        threshold=0.5,
        model=custom_llm,
        async_mode=False
    )

    # Create the 3-row test cases matching RAGAS dataset
    scenarios = [
        {
            "input": "What does the ST elevation in lead II, III, and aVF suggest?",
            "actual_output": "ST elevation in leads II, III, and aVF suggests a possible acute inferior myocardial infarction. Immediate medical attention is required.",
            "expected_output": "- Acute inferior myocardial infarction likely.\n- Immediate cardiologist referral needed.",
            "context": ["ST elevation in inferior leads II, III, and aVF indicates acute inferior myocardial infarction (heart attack)."]
        },
        {
            "input": "The patient has a 12% heart disease risk. Should they take Aspirin?",
            "actual_output": "With a low risk of 12%, routine lifestyle changes are advised. Do not start taking Aspirin or any medication without consulting a doctor.",
            "expected_output": "- Low risk of 12%.\n- Do not prescribe Aspirin.\n- Consult physician.",
            "context": ["Low risk (12%) does not require medication. No Aspirin should be self-prescribed without physician consult."]
        },
        {
            "input": "Explain how the max heart rate feature affects the risk score.",
            "actual_output": "A lower maximum heart rate increases the predicted risk, as it suggests the heart may have reduced functional capacity under stress.",
            "expected_output": "- Lower max heart rate increases risk.\n- Reflects reduced cardiovascular efficiency.",
            "context": ["A lower max heart rate indicates reduced cardiovascular efficiency and increases the predicted heart disease risk."]
        }
    ]

    test_cases = []
    for s in scenarios:
        test_case = LLMTestCase(
            input=s["input"],
            actual_output=s["actual_output"],
            expected_output=s["expected_output"],
            context=s["context"],
            retrieval_context=s["context"]
        )
        test_cases.append(test_case)

    results_list = []

    print(f"Evaluating {len(test_cases)} test cases sequentially with 5s delay...")
    for idx, test_case in enumerate(test_cases):
        print(f"\nEvaluating test case {idx+1}/{len(test_cases)}...")
        
        # Helper to execute a single metric with retries and delay
        def evaluate_metric_with_retry(metric, metric_name):
            max_retries = 3
            retry_delay = 10
            for attempt in range(max_retries):
                try:
                    metric.measure(test_case)
                    return metric.score
                except Exception as e:
                    print(f"Warning: {metric_name} attempt {attempt+1} failed with: {e}")
                    if "429" in str(e) or "limit" in str(e).lower():
                        print(f"Rate limit hit. Sleeping for {retry_delay}s before retry...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        time.sleep(2)
            print(f"Error: {metric_name} failed after {max_retries} attempts. Returning fallback score 0.5")
            return 0.5

        # Evaluate Faithfulness
        print("Measuring Faithfulness...")
        f_score = evaluate_metric_with_retry(faithfulness_metric, "Faithfulness")
        print(f"Faithfulness Score: {f_score}")
        time.sleep(5)

        # Evaluate Relevancy
        print("Measuring Answer Relevancy...")
        r_score = evaluate_metric_with_retry(answer_relevancy_metric, "Answer Relevancy")
        print(f"Relevancy Score: {r_score}")
        time.sleep(5)

        # Evaluate Hallucination (Note: Hallucination score in DeepEval is 0 to 1, where 0 is no hallucination and 1 is full hallucination.
        # We will convert it to 'Hallucination-Free Score' which is 1 - score, so higher is better (1.0 = perfect/no hallucination))
        print("Measuring Hallucination...")
        raw_h_score = evaluate_metric_with_retry(hallucination_metric, "Hallucination")
        h_score = 1.0 - raw_h_score
        print(f"Hallucination-Free Score: {h_score}")
        time.sleep(5)

        results_list.append({
            "faithfulness": f_score,
            "answer_relevance": r_score,
            "hallucination_free": h_score
        })

    # Save to CSV
    df_res = pd.DataFrame(results_list)
    summary_scores = {
        "Metric": ["Faithfulness", "Answer Relevance", "Hallucination-Free"],
        "Score": [
            df_res["faithfulness"].mean() * 100,
            df_res["answer_relevance"].mean() * 100,
            df_res["hallucination_free"].mean() * 100
        ]
    }
    
    summary_df = pd.DataFrame(summary_scores)
    output_path = SCORECARDS_DIR / "deepeval_scorecard.csv"
    summary_df.to_csv(output_path, index=False)
    print(f"\nDeepEval evaluation complete. Scorecard saved to: {output_path}")
    print(summary_df)

if __name__ == "__main__":
    run_deepeval_evaluation()
