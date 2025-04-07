import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# Load the dataset
df = pd.read_csv("infilled_stories.csv")
df = df.rename(columns={
    "0_infill": "Infills: 0",
    "1_infill": "Infills: 1",
    "4_infill": "Infills: 4",
    "16_infill": "Infills: 16",
    "64_infill": "Infills: 64"
})
# Define model parameters
MODEL_NAME = "gpt2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 3
TOP_K = 5
TEMPERATURES = [0.01, 0.5, 1.0, 1.5, 2.0]
INFILL_TYPES = ["Infills: 0", "Infills: 1", "Infills: 4", "Infills: 16", "Infills: 64"]

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

def generate_next_tokens_huggingface(prompt, temperature):
    """
    Generate tokens step-by-step, showing top-k probabilities.
    """
    current_tokens = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    if current_tokens.shape[1] >= 1024:
        return None  # Skip if token length exceeds 1024
    
    all_probs = []
    for step in range(MAX_NEW_TOKENS):
        with torch.no_grad():
            outputs = model(current_tokens)

        logits = outputs.logits[0, -1, :]
        logits /= temperature
        probs = torch.softmax(logits, dim=-1)

        top_probs, top_indices = torch.topk(probs, TOP_K)
        top_probs = top_probs.cpu().numpy().tolist()
        top_indices = top_indices.cpu().numpy().tolist()
        top_tokens = [tokenizer.decode([idx]) for idx in top_indices]

        all_probs.append((step, top_tokens, top_probs))
        best_token_id = top_indices[0]
        current_tokens = torch.cat([current_tokens, torch.tensor([[best_token_id]]).to(DEVICE)], dim=1)
    
    return all_probs

def plot_token_probabilities(all_probs, ax, title, is_first_column, is_last_row):
    if all_probs is None:
        ax.text(0.5, 0.5, "Question Skipped", fontsize=16, ha='center', va='center', color='red')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        return
    
    num_steps = len(all_probs)
    max_rank = max(len(step[2]) for step in all_probs)
    avg_probs = [[] for _ in range(max_rank)]
    token_labels = [[] for _ in range(max_rank)]
    
    for step, top_tokens, top_probs in all_probs:
        for rank in range(len(top_probs)):
            avg_probs[rank].append(top_probs[rank])
            token_labels[rank].append(top_tokens[rank])
    
    avg_probs = [np.mean(probs) if probs else 0 for probs in avg_probs]
    x_labels = ["CT", "1AP", "2AP", "3AP", "4AP"][:len(avg_probs)]
    
    token_labels_str = []
    if token_labels:
        for label, tokens in zip(x_labels, token_labels):
            if tokens:
                escaped_tokens = [token.replace('$', r'\$') for token in tokens]
                token_labels_str.append(f"{label}: {', '.join(escaped_tokens)}")
    
    bars = ax.bar(x_labels, avg_probs, color='darkblue', alpha=0.7)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height,
                f'{height:.2f}', ha='center', va='bottom', color='blue', fontsize=17,fontweight='bold')
    
    if token_labels_str:
        token_text = "\n".join(token_labels_str)
        ax.text(0.98, 0.98, token_text, fontsize=16, verticalalignment='top', horizontalalignment='right',
                transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))
    
    # Display 'Average Probability' in the first column
    if is_first_column:
        ax.set_ylabel("Average Probability",fontweight='bold',fontsize=20)
    else:
        ax.set_ylabel("")
    
    # Show x_labels only in the last row
    if is_last_row:
        ax.set_xlabel("Token Category",fontweight='bold',fontsize=20)
        ax.set_xticklabels(x_labels, fontsize=20, rotation=45,fontweight='bold')  # Display labels in the last row only
    else:
        ax.set_xticks([])  # Remove x-ticks for other subplots
    if not is_first_column:
        ax.set_yticks([])

    ax.set_ylim(0, 1.1)
    ax.tick_params(axis='y', labelsize=16)

   
    ax.set_title(title, fontsize=20, fontweight='bold')

pdf_filename = "Users/pavan.yadav/exp/gpt2_so_5march.pdf"
with PdfPages(pdf_filename) as pdf:
    for idx, row in df.iterrows():
        fig, axs = plt.subplots(len(TEMPERATURES), len(INFILL_TYPES), figsize=(30, 26))
        plt.subplots_adjust(wspace=0.1, hspace=0.2)
        
        for temp_idx, temp in enumerate(TEMPERATURES):
            for infill_idx, infill_type in enumerate(INFILL_TYPES):
                if infill_type in df.columns and pd.notna(row[infill_type]) and pd.notna(row["so_question"]):
                    prompt = row[infill_type] + " " + row["so_question"]
                    all_probs = generate_next_tokens_huggingface(prompt, temp)
                    
                    # Check if it is the first column and last row
                    is_first_column = (infill_idx == 0)
                    is_last_row = (temp_idx == len(TEMPERATURES) - 1)
                    
                    plot_token_probabilities(all_probs, axs[temp_idx, infill_idx],
                                             f"Temp: {temp}, {infill_type}",
                                             is_first_column, is_last_row)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

print(f"Plots saved in {pdf_filename}")
