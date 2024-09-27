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

## Prompt Design

The code generation process leverages specifically designed prompts to optimize the performance and accuracy of the models. Below are examples of the prompt templates used for generating CNNs and ViTs:

- **Prompt 1 Template:**
  > "Generate a TensorFlow script to classify the MNIST dataset into 10 classes using [architecture type]."

- **Prompt 2 Template for Time Optimization:**
  > "Create a TensorFlow script for classifying 10 sign classes from the MNIST dataset using [architecture type]. Focus on optimizing the code to minimize training time."

- **Prompt 3 Template for High Validation Accuracy:**
  > "Create a TensorFlow script for classifying 10 sign classes from the MNIST dataset using [architecture type]. Focus on optimizing the code to achieve high testing accuracy."

Replace `[architecture type]` with either **"Convolutional Neural Network (CNN)"** or **"Vision Transformer (ViT)"** as appropriate.

## Usage

To utilize the provided scripts:

1. Clone the repository to your local machine.
2. Navigate to the desired architecture folder.
3. Execute the TensorFlow scripts in an appropriate Python environment.

## Acknowledgments

We would like to acknowledge the LLMs used in this study and the datasets that made this research possible.
