import pandas as pd
import os
import re
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_openai import AzureChatOpenAI

# Load environment variables
load_dotenv()
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("API_VERSION")
api_key = os.getenv("AZURE_OPENAI_API_KEY")

# Initialize Azure GPT-4 model
llm = AzureChatOpenAI(
    model="gpt-4",
    temperature=0.7,  # Allow slight creativity
    azure_endpoint=azure_endpoint,
    api_key=api_key,
    max_tokens=1024,  # Increase token limit for longer infills
    api_version=api_version
)

# Load the dataset
input_csv = "original_stories.csv"  # Update with actual path
df = pd.read_csv(input_csv)

# Sentence counting function
def count_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return len(sentences)

# Function to generate additional sentences
def generate_additional_sentences(base_story, num_sentences):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("You are a creative AI that expands stories by adding coherent and relevant sentences."),
        HumanMessagePromptTemplate.from_template(
            f"Expand the following story by adding exactly {num_sentences} new sentences:\n\n{base_story}\n\nExtended story:"
        )
    ])

    chain = LLMChain(llm=llm, prompt=prompt)
    generated_story = chain.run({})

    # Validate the generated output
    sentences = re.split(r'(?<=[.!?])\s+', generated_story.strip())
    expected_count = count_sentences(base_story) + num_sentences

    # Trim or retry if needed
    if len(sentences) > expected_count:
        generated_story = " ".join(sentences[:expected_count])
    elif len(sentences) < expected_count:
        print(f"Warning: Expected {expected_count} sentences, but got {len(sentences)}")

    return generated_story

# Process each story
infill_data = []
for _, row in df.iterrows():
    original_story = row['original_story']

    # Generate infills in a stepwise manner
    infills = {}
    infills[1] = generate_additional_sentences(original_story, 1)  # 1 new sentence
    infills[4] = generate_additional_sentences(infills[1], 3)  # 3 more sentences
    infills[16] = generate_additional_sentences(infills[4], 12)  # 12 more sentences
    infills[64] = generate_additional_sentences(infills[16], 48)  # 48 more sentences

    # Validate sentence count
    for level, added in [(1, 1), (4, 4), (16, 16), (64, 64)]:
        expected_sentences = count_sentences(original_story) + added
        actual_sentences = count_sentences(infills[level])
        if actual_sentences != expected_sentences:
            print(f"Error in {level}-infill: Expected {expected_sentences} sentences, but got {actual_sentences}")

    # Store results
    infill_data.append({
        'original_story': original_story,
        '1_infill': infills[1],
        '4_infill': infills[4],
        '16_infill': infills[16],
        '64_infill': infills[64]
    })

# Save the results
output_csv = "infilled_stories.csv"
pd.DataFrame(infill_data).to_csv(output_csv, index=False)

print(f"Infilled stories saved to {output_csv}")
