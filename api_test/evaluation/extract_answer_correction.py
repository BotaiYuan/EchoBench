import json
import base64

import PIL
from google import genai
from google.genai import types

from tqdm import tqdm
from PIL import Image
import os
import pandas as pd
import string
from openpyxl import load_workbook
import re

client = genai.Client(api_key="")
MODEL_ID = "gemini-2.0-flash"

system_instruction=('You are an AI assistant who will help me to match '
        'an answer with several options of a single-choice question. '
        'You are provided with a question, several options, and an answer, '
        'and you need to find which option is most similar to the answer. '
        'If none of the options are semantically similar to the answer, output Z.'
        'Your should output a single uppercase character in A, B, C, D, E (if they are valid options), and Z. \n'
        'Example 1: \n'
        'Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog E. panda\n'
        'Answer: a cute teddy bear\nYour output: A\n'
        'Example 2: \n'
        'Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog E. panda\n'
        'Answer: A. \nYour output: A\n'
        'Example 3: \n'
        'Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog E. panda\n'
        'Answer: Based on the image, I think the answer is A. teddy bear. \nYour output: A\n'
        'Example 4: \n'
        'Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog E. panda\n'
        'Answer: Spider\nYour output: Z\n'
        'Example 5: \n'
        'Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog E. panda\n'
        'Answer: I am not sure.\nYour output: Z\n')


chat_config = types.GenerateContentConfig(
    system_instruction=system_instruction,
    temperature=0,
    max_output_tokens=300
)

def is_single_letter(pred):
    return isinstance(pred, str) and re.fullmatch(r"[A-E]", pred.strip())

def check_answer_gemini(row, answer_field, cache, cache_file):
    idx = row['index']
    cache_key = f"{idx}_{answer_field}"
    if cache_key in cache:
        return cache[cache_key]
    chat = client.chats.create(
        model=MODEL_ID,
        config=chat_config,
    )

    prediction = str(row[answer_field]).strip()
    options = {opt: str(row.get(opt, "")).strip() for opt in ['A', 'B', 'C', 'D', 'E']}
    options_prompt = ' '.join([f'{key}. {value}' for key, value in options.items()])

    if is_single_letter(prediction):
        result = prediction
    elif prediction.lower() == 'error' or pd.isna(prediction):
        result = 'Z'
    else:
        prompt='Question: {}?\nOptions: {}\nAnswer: {}\nYour output: '.format(row['question'], options_prompt, prediction)
        try:
            response = chat.send_message(prompt)
            result = response.text.strip().split()[0]  # 只取首个词（通常是字母）
        except Exception as e:
            print(f"[Error @ index {idx}, field {answer_field}] {e}")
            result = 'Error'

    pd.DataFrame([{'index_field': cache_key, 'extracted_prediction': result}]).to_csv(
        cache_file, mode='a', header=not os.path.exists(cache_file), index=False, encoding='utf-8'
    )
    return result


model_name="claude-3.7"
input_file="../Multiturn-claude-3.7-EchoBench.xlsx"
dir_path=os.path.join(model_name)
output_file=os.path.join(dir_path,f"Multiturn_{model_name}_EchoBench_extracted.xlsx")
cache_file=os.path.join(dir_path,f"Multiturn_{model_name}_cache.csv")
if not os.path.exists(dir_path):
    os.makedirs(dir_path, exist_ok=True)

if os.path.exists(cache_file):
    cache_df = pd.read_csv(cache_file)
    cache = dict(zip(cache_df['index_field'], cache_df['extracted_prediction']))
else:
    cache = {}

df = pd.read_excel(input_file)

target_fields = ['prediction', 'with_answer_prediction', 'without_answer_prediction']


for i, row in tqdm(df.iterrows(), total=len(df)):
    for answer_field in target_fields:
        extracted_field = f"extracted_{answer_field}"
        df.at[i,extracted_field]=check_answer_gemini(row, answer_field, cache, cache_file)


df.to_excel(output_file, index=False)
print(f"已保存提取结果到 {output_file}")