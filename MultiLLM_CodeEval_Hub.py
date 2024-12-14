import openai
import google.generativeai as genai
import streamlit as st
import os
from llamaapi import LlamaAPI  # Import the LLaMA SDK
import re

# Set API keys from environment variables or configuration files for security
openai.api_key = os.getenv('OPENAI_API_KEY')  # Ensure OPENAI_API_KEY is set in environment variables
genai.configure(api_key=os.getenv('GENAI_API_KEY'))  # Ensure GENAI_API_KEY is set in environment variables
LLAMA_API_KEY = os.getenv('LLAMA_API_KEY')  # Ensure LLAMA_API_KEY is set in environment variables

# Initialize the LLaMA API
llama = LlamaAPI(LLAMA_API_KEY)

# Function to generate code using ChatGPT
def generate_chatgpt_code(task, dataset):
    """
    Generate TensorFlow code using OpenAI's ChatGPT model for a given task and dataset.
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": f"Generate TensorFlow code for {task} using {dataset}."}]
    )
    return response["choices"][0]["message"]["content"].strip()

# Function to generate code using Gemini
def generate_gemini_code(task, dataset):
    """
    Generate TensorFlow code using Google's Gemini model for a given task and dataset.
    """
    response = genai.GenerativeModel(model_name="gemini-1.5-flash").generate_content(
        f"Generate TensorFlow code for {task} using {dataset}."
    )
    return response.text.strip()

# Function to generate code using LLaMA
def generate_llama_code(task, dataset):
    """
    Generate TensorFlow code using LLaMA model for a given task and dataset.
    """
    api_request_json = {
        "model": "llama3.1-70b",  # Replace with the correct model name if needed
        "messages": [
            {"role": "user", "content": f"Generate TensorFlow code for {task} using {dataset}."},
        ],
        "stream": False,
        "max_tokens": 2000  # Increased max tokens to print full code
    }

    response = llama.run(api_request_json)
    full_output = response.json()["choices"][0]["message"]["content"]
    return full_output.strip()


# Prompts for evaluation
PROMPT_P1 = """
Deep Learning Code Evaluation:
Evaluate the deep learning code for a classification task across various dimensions. 
Consider the following metrics:
- Data Augmentation: Score 1-5 (No augmentation, Basic augmentation, Advanced augmentation, Comprehensive augmentation, Expert-level augmentation)
- Model Architecture: Score 1-5 (Simple architecture, Basic architecture, Advanced architecture, Comprehensive architecture, Expert-level architecture)
- Training: Score 1-5 (Insufficient Training, Basic Training, Moderate Training, Comprehensive Training, Expert-Level Training)
- Hyperparameter Tuning: Score 1-5 (No tuning, Basic tuning, Advanced tuning, Comprehensive tuning, Expert-level tuning)
- Evaluation Metrics: Score 1-5 (Limited metrics, Basic metrics, Advanced metrics, Comprehensive metrics, Expert-level metrics)
Provide a summary table with scores and explanations for each metric.
Also, provide the gaps in the code and suggest improvements.
"""

PROMPT_P2 = """
Code Readability and Maintainability Evaluation:
Evaluate the readability and maintainability of the deep learning code. 
Consider the following metrics:
- Code Organization: Score 1-5 (Poor organization, Basic organization, Modular organization, Clear organization, Well-documented organization)
- Variable Naming: Score 1-5 (Unclear naming, Basic naming, Descriptive naming, Consistent naming, Standardized naming)
- Commenting: Score 1-5 (No comments, Basic comments, Detailed comments, Clear explanations, Interactive comments)
- Error Handling: Score 1-5 (No error handling, Basic error handling, Advanced error handling, Robust error handling, Automated error handling)
- Code Reusability: Score 1-5 (Low reusability, Moderate reusability, High reusability, Modular reusability, Library-level reusability)
Provide a summary table with scores and explanations for each metric, and also explain the gaps and improvement suggestions.
"""

# Function to evaluate code using OpenAI API
def evaluate_code(prompt, code, model="gpt-4"):
    """
    Evaluate the deep learning code for quality, maintainability, and effectiveness using OpenAI's GPT model.
    """
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "system", "content": "You are an expert code evaluator."},
                  {"role": "user", "content": prompt + f"\n\nCode:\n{code}"}],
        max_tokens=1000,
        temperature=0.7
    )
    return response['choices'][0]['message']['content'].strip()

# Function to extract scores and improvement suggestions from evaluation text
def extract_scores_and_improvements(evaluation_text):
    """
    Extract scores and generate improvement suggestions based on the evaluation feedback.
    """
    scores = {}
    improvements = []

    # Metrics to extract scores for
    metrics = [
        "Data Augmentation", "Model Architecture", "Training", 
        "Hyperparameter Tuning", "Evaluation Metrics",
        "Code Organization", "Variable Naming", "Commenting", 
        "Error Handling", "Code Reusability"
    ]

    # Regex to extract scores from evaluation text
    for metric in metrics:
        match = re.search(rf"{metric}.*?(\d)", evaluation_text)
        if match:
            score = int(match.group(1))
            scores[metric] = score

    # Generate improvement suggestions based on scores
    for metric, score in scores.items():
        if score < 3:
            improvements.append(f"Enhance {metric.lower()} to improve code quality.")

    gaps_improvements_text = "\n".join(improvements) if improvements else "No specific gaps found; code is satisfactory."

    return scores, gaps_improvements_text

# Function to extract total score from evaluation text
def extract_total_score(evaluation_text):
    """
    Extract the total score by summing the individual metric scores from the evaluation feedback.
    """
    total_score = 0
    metrics = [
        "Data Augmentation", "Model Architecture", "Training", 
        "Hyperparameter Tuning", "Evaluation Metrics",
        "Code Organization", "Variable Naming", "Commenting", 
        "Error Handling", "Code Reusability"
    ]
    
    for metric in metrics:
        match = re.search(rf"{metric}.*?(\d)", evaluation_text)
        if match:
            score = int(match.group(1))
            total_score += score

    return total_score

def main():
    """
    Main function to drive the Streamlit interface, generate code using various LLMs, and evaluate the results.
    """
    st.title("MultiLLM CodeEval Hub")

    task = st.text_input("Enter the task (e.g., image classification):")
    dataset = st.text_input("Enter the dataset (e.g., MNIST):")

    if st.button("Generate and Evaluate Code"):
        llms = ["ChatGPT", "Gemini", "LLaMA"]  # Added LLaMA to the list of LLMs
        results = {}

        for llm in llms:
            try:
                st.write(f"Generating code with {llm}...")

                # Generate code using the respective LLM
                if llm == "ChatGPT":
                    code = generate_chatgpt_code(task, dataset)
                elif llm == "Gemini":
                    code = generate_gemini_code(task, dataset)
                elif llm == "LLaMA":
                    code = generate_llama_code(task, dataset)

                # Display generated code in Streamlit UI
                st.code(code, language="python")

                # Evaluate the generated code with both prompts
                st.write(f"Evaluating code generated by {llm}...")

                # Evaluate with both prompts
                evaluation_text_p1 = evaluate_code(PROMPT_P1, code)
                evaluation_text_p2 = evaluate_code(PROMPT_P2, code)

                # Display evaluation feedback for both prompts directly
                st.write(f"### Evaluation for Deep Learning Code (by {llm}):")
                st.markdown(evaluation_text_p1)

                st.write(f"### Evaluation for Code Readability and Maintainability (by {llm}):")
                st.markdown(evaluation_text_p2)

                # Store the results for comparison later
                results[llm] = {
                    "evaluation_p1": evaluation_text_p1,
                    "evaluation_p2": evaluation_text_p2
                }

            except Exception as e:
                st.write(f"Error occurred for {llm}: {e}")

        # After determining the best LLM, display its generated code
        if results:
            total_scores = {}
            best_llm_code = ""

            for llm, evaluations in results.items():
                # Extract total score from both evaluations (p1 and p2)
                total_score_p1 = extract_total_score(evaluations["evaluation_p1"])
                total_score_p2 = extract_total_score(evaluations["evaluation_p2"])

                # Calculate total score by summing both evaluation scores
                total_score = total_score_p1 + total_score_p2
                total_scores[llm] = total_score

                # Store the code generated by the best LLM
                if total_score == max(total_scores.values()):
                    best_llm_code = evaluations["evaluation_p1"]  # Or you can choose the generated code directly

            # Find the best LLM based on the total score
            best_llm = max(total_scores, key=total_scores.get)
            best_llm_score = total_scores[best_llm]

            # Display the best LLM with the total score
            st.write(f"**Best LLM: {best_llm} - Total Score: {best_llm_score}**")
            
            # Display the best LLM generated code
            st.write(f"### Code generated by {best_llm}:")
            st.code(best_llm_code, language="python")


if __name__ == "__main__":
    main()
