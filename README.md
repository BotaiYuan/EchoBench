# EchoBench

Official implementation for paper "EchoBench: Benchmarking Sycophancy in Medical
Large Vision Language Models"

Before start, please download "EchoBench.tsv" from our official huggingface repository.

============open-source_models============

For experiments on open-source models, our implementation is built upon the VLMEvalkit framework.

1. Navigate to the VLMEval directory

2. Set up the environment by running: "pip install -e ."

3. Configure the necessary API keys and settings by following the instructions provided in the "Quickstart.md" file of VLMEvalkit.

4. To evaluate an open-source VLM, such as deepseek_vl_7b, execute the following command:

AUTO_SPLIT=1 torchrun --nproc-per-node=1 run.py --model deepseek_vl_7b --data EchoBench --verbose --mode infer

The output will be saved to "deepseek_vl_7b_EchoBench. xlsx"

5. Navigate to the "evaluation" directory and run: "extract_answer.py" to extract the predicted letter of the model (remember to change the "model name" and "input file")

============proprietary_models============

1. Navigate to the api_test directory

2. Run the script: "python localize_dataset.py" （Remember to give the path to EchoBench.tsv)

3. Execute the corresponding Python script for each model. Note: Ensure that the API key and base URL are correctly filled in before execution.
   
4. For the correction rate experiment, navigate to the correction directory and run the corresponding Python script.
   
5. Navigate to the "evaluation" directory and run: "extract_answer.py" to extract the predicted letter of each model

6. Execute "statistics.py" and "statistics_correction.py" to get the performance metric of each model


