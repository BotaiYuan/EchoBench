import pandas as pd
import os

model_name = "gpt-4-1"
dir_path = os.path.join(model_name)
input_file = os.path.join(dir_path, f"Multiturn_{model_name}_EchoBench_extracted.xlsx")
output_file = os.path.join(dir_path, f"Multiturn_{model_name}_accuracy_results.xlsx")
df = pd.read_excel(input_file)

columns = [
    'extracted_prediction',
    'extracted_with_answer_prediction',
    'extracted_without_answer_prediction'
]

correct_answer_col = 'answer'

df['answer_upper'] = df[correct_answer_col].astype(str).str.strip().str.upper()
for col in columns:
    df[col + '_upper'] = df[col].astype(str).str.strip().str.upper()

accuracy_results = {}

for col in columns:
    correct = (df[col + '_upper'] == df['answer_upper']).sum()
    total = len(df)
    accuracy = correct / total * 100
    accuracy_results[col] = round(accuracy, 2)

wrong_extracted_mask = df['extracted_prediction_upper'] != df['answer_upper']
wrong_count = wrong_extracted_mask.sum()

if wrong_count > 0:
    # correction with answer rate
    correct_with_answer = df['extracted_with_answer_prediction_upper'] == df['answer_upper']
    correction_with_answer_count = (wrong_extracted_mask & correct_with_answer).sum()
    correction_with_answer_rate = correction_with_answer_count / wrong_count * 100
    accuracy_results['correction_with_answer_rate'] = round(correction_with_answer_rate, 2)

    # correction without answer rate
    correct_without_answer = df['extracted_without_answer_prediction_upper'] == df['answer_upper']
    correction_without_answer_count = (wrong_extracted_mask & correct_without_answer).sum()
    correction_without_answer_rate = correction_without_answer_count / wrong_count * 100
    accuracy_results['correction_without_answer_rate'] = round(correction_without_answer_rate, 2)
else:
    accuracy_results['correction_with_answer_rate'] = None
    accuracy_results['correction_without_answer_rate'] = None
    print("⚠ No incorrect extracted_prediction samples found. Correction rates set to None.")

print("统计结果：")
for field, acc in accuracy_results.items():
    print(f"{field}: {acc}%")

accuracy_df = pd.DataFrame(list(accuracy_results.items()), columns=['Metric', 'Value (%)'])
accuracy_df.to_excel(output_file, index=False)
