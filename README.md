# Towards Autonomous Deep Learning: Comparative Analysis of AI-Generated and AI-Evaluated Code Using LLMs for Image Classification
## Repository Overview

This repository contains the code for the research paper titled **"Towards Autonomous Deep Learning: Comparative Analysis of AI-Generated and AI-Evaluated Code Using LLMs for Image Classification."** The main objective of this research is to explore the capabilities of Large Language Models (LLMs) in generating code for deep learning models, specifically for image classification tasks. The models evaluated include **Convolutional Neural Networks (CNNs)** and **Vision Transformers (ViTs)**, generated using various LLMs such as **ChatGPT, Copilot, Gemini,** and **LLaMA**.

## Directory Structure

The repository is organized into three main folders, each dedicated to a different architecture:

### 1. Convolutional Neural Network

- Contains TensorFlow code files named in the format `P1_ChatGPT_TF`, `P2_LLAMA_TF`, etc. where:
  - `P1`, `P2`, and `P3` denote different prompt types.
  - The second part indicates the tool used for code generation (e.g., ChatGPT, LLaMA, etc.).
  - The suffix `TF` indicates that TensorFlow is utilized for the implementation.

### 2. Vision Transformer

- Similar structure as the CNN folder, with code files generated for ViTs.

### 3. Transfer Learning

- Contains code related to transfer learning approaches, also structured similarly.

### 4. Open_AI_DL_Eval.py

- The `Open_AI_DL_Eval.py` file is located in the main directory of the repository. This script automates the evaluation of the deep learning code generated by different LLMs using the OpenAI API.
- The evaluation focuses on two main aspects:
  - Deep learning code evaluation
  - Code readability and maintainability evaluation
- The code produced by the tools-ChatGPT, LLaMA, Gemini, and Copilot-is assessed via this Python script.

#### Configuration Sections

Before running the script, please update the following sections with your respective paths:

```python
openai.api_key = 'Your_API_Key'  # Set your OpenAI API key
folder_path = r"Your_PY_Files_Path"  # Specify the path to your Python files
output_file = os.path.join(folder_path, r'Your_Output_Excel_Path')  # Set the output path for the Excel file
