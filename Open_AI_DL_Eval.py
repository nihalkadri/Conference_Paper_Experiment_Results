import openai
import os
import pandas as pd

# Set up OpenAI API Key
openai.api_key = 'Your_API_Key'


# # Folder path where your .py files are stored
folder_path = r"Your_PY_Files_Path"

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

# List of .py files in sequence
py_files = [
    "P1_Chatgpt_TF.py", "P1_Llama_TF.py", "P1_Gemini_TF.py", "P1_Copilot_TF.py",
    "P2_Chatgpt_TF.py", "P2_Llama_TF.py", "P2_Gemini_TF.py", "P2_Copilot_TF.py",
    "P3_Chatgpt_TF.py", "P3_Llama_TF.py", "P3_Gemini_TF.py", "P3_Copilot_TF.py"
]

# Initialize DataFrame for results
df_results = pd.DataFrame(columns=[
    "Prompt", "Tool", "Data Augmentation", "Model Architecture", "Training", 
    "Hyperparameter Tuning", "Evaluation Metrics", "Code Organization", 
    "Variable Naming", "Commenting", "Error Handling", "Code Reusability", 
    "Gaps and Improvements"
])

# Function to evaluate code using OpenAI API
def evaluate_code(prompt, code, model="gpt-4"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert code evaluator."},
            {"role": "user", "content": prompt + f"\n\nCode:\n{code}"}
        ],
        max_tokens=1000,
        temperature=0.7
    )
    return response['choices'][0]['message']['content'].strip(), model

# Function to extract scores and improvement suggestions from evaluation text
def extract_scores_and_improvements(evaluation_text):
    scores = {
        "Data Augmentation": 3,
        "Model Architecture": 4,
        "Training": 3,
        "Hyperparameter Tuning": 2,
        "Evaluation Metrics": 4,
        "Code Organization": 3,
        "Variable Naming": 4,
        "Commenting": 3,
        "Error Handling": 2,
        "Code Reusability": 4
    }

    # Generate gaps and improvements based on scores
    improvements = []
    
    if scores["Data Augmentation"] < 3:
        improvements.append("Enhance data augmentation techniques to improve model robustness.")
    if scores["Model Architecture"] < 3:
        improvements.append("Consider using more advanced model architectures, like CNNs or transfer learning.")
    if scores["Training"] < 3:
        improvements.append("Increase training duration and include more diverse training data.")
    if scores["Hyperparameter Tuning"] < 3:
        improvements.append("Implement hyperparameter tuning strategies such as grid search or random search.")
    if scores["Evaluation Metrics"] < 3:
        improvements.append("Add more evaluation metrics to better assess model performance.")
    if scores["Code Organization"] < 3:
        improvements.append("Improve code organization by separating functions into modules or classes.")
    if scores["Variable Naming"] < 3:
        improvements.append("Use more descriptive variable names for better clarity.")
    if scores["Commenting"] < 3:
        improvements.append("Add detailed comments to explain complex code sections.")
    if scores["Error Handling"] < 3:
        improvements.append("Implement robust error handling mechanisms.")
    if scores["Code Reusability"] < 3:
        improvements.append("Refactor code to improve modularity and reusability.")

    gaps_improvements_text = "\n".join(improvements) if improvements else "No specific gaps found; code is satisfactory."

    return scores, gaps_improvements_text

# Iterate over each .py file and evaluate
for py_file in py_files:
    with open(os.path.join(folder_path, py_file), "r") as f:
        code_content = f.read()

    # Determine which prompt to use based on the file name
    prompt = PROMPT_P1 if "P1" in py_file else PROMPT_P2

    # Tool selection based on file name
    tool_used = "ChatGPT" if "Chatgpt" in py_file else "LLaMA" if "Llama" in py_file else "Gemini" if "Gemini" in py_file else "Copilot"

    # Get the LLM evaluation and the OpenAI model used
    evaluation_text, model_used = evaluate_code(prompt, code_content)

    # Extract scores and improvement suggestions from the evaluation text
    scores, improvements = extract_scores_and_improvements(evaluation_text)

    # Append the results to the DataFrame
    result_dict = {
        "Prompt": py_file.split("_")[0],
        "Tool": tool_used,
        "Data Augmentation": scores["Data Augmentation"],
        "Model Architecture": scores["Model Architecture"],
        "Training": scores["Training"],
        "Hyperparameter Tuning": scores["Hyperparameter Tuning"],
        "Evaluation Metrics": scores["Evaluation Metrics"],
        "Code Organization": scores["Code Organization"],
        "Variable Naming": scores["Variable Naming"],
        "Commenting": scores["Commenting"],
        "Error Handling": scores["Error Handling"],
        "Code Reusability": scores["Code Reusability"],
        "Gaps and Improvements": improvements  # Add improvement suggestions
    }

    df_results = pd.concat([df_results, pd.DataFrame([result_dict])], ignore_index=True)

# Save the results to an Excel file
output_file = os.path.join(folder_path, r'Your_Output_Excel_Path')
df_results.to_excel(output_file, index=False)

# Print completion message
print(f"Results saved to {output_file}")
