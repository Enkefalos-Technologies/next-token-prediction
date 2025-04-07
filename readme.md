# Exploring Next Token Prediction in Theory of Mind (ToM) Task: A Comparative Experiment with GPT-2 and LLaMA-2 Models

## 🧠 Project Overview

This project explores the **Theory of Mind (ToM)** through NLP models by predicting next tokens and analyzing how different models—**GPT-2** and **LLaMA-2**—perform on **first-order**, **zero-order**, and **second-order** questions.

---

## 📁 Project Structure

### 🔹 Data
- `Original_stories.csv` → Contains 10 original stories from the *Explore Theory of Mind* paper.
- `infilled_stories.csv` → Infills generated using `infill_generator_usinggpt4.py`.

### 🔹 Code
- `python infill_generator_usinggpt4.py` → Run the script file to generate infilled stories.
- `python gpt2_model_pred.py` → Run this script file to predicts next tokens using GPT-2 and plots graphs.
- `python llama2_model_pred.py` → Run this script file to predicts next tokens using LLaMA-2 and plots graphs.

---

## 🧰 Requirements

- `requirements.txt` → Contains all the necessary dependencies.

Install dependencies using:

```bash
!pip install -r requirements.txt
