🚀 Exploring Next Token Prediction in
Theory of Mind (ToM) Task: A
Comparative Experiment with GPT-2 and
Llama-2 Models

📌 Project Overview

This project explores the Theory of Mind using NLP models by predicting next tokens and analyzing how different models perform on first-order, zero-order, and second-order questions.

📂 Project Structure

🔹 Data

Original_stories.csv → Contains 10 original stories from the Explore Theory of Mind paper.

infilled_stories.csv → Infills generated using infill_generator_usinggpt4.py.

🔹 Code

infill_generator_usinggpt4.py → Script to generate infilled stories.

gpt2_model_pred.py → Predicts next tokens using GPT-2 and plots graphs.

llama2_model_pred.py → Predicts next tokens using LLaMA-2 and plots graphs.

🔹 Requirements

requirements.txt → Contains necessary dependencies.

🔹 Results (Plotted Graphs)

📊 GPT-2 Model Predictions:

gpt2_fo_5march.pdf → First-order question results.

gpt2_so_5march.pdf → Second-order question results.

gpt2_zo_5march.pdf → Zero-order question results.

📊 LLaMA-2 Model Predictions:

llama2_fo_5march.pdf → First-order question results.

llama2_so_5march.pdf → Second-order question results.

llama2_zo_5march.pdf → Zero-order question results.

🛠 Installation

To set up the project, follow these steps:

# Clone the repository
git clone https://github.com/Enkefalos-Technologies/next-token-prediction.git
cd your-repo-name

# Install dependencies
pip install -r requirements.txt

🚀 Usage

Run the scripts in the following order:

# Generate infilled stories
python infill_generator_usinggpt4.py or use the  infilled_stories.csv file

# Run GPT-2 model predictions
python gpt2_model_pred.py

# Run LLaMA-2 model predictions
python llama2_model_pred.py

📜 License

This project is open-source and available under the MIT License.

![GPT-2 First Order Prediction](gpt2_fo_page1.png)