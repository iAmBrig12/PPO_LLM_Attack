sst2
textattack/bert-base-uncased-SST-2  
textattack/roberta-base-SST-2
textattack/distilbert-base-uncased-SST-2

mrpc
textattack/bert-base-uncased-MRPC
textattack/roberta-base-MRPC
textattack/distilbert-base-uncased-MRPC

mnli 
textattack/bert-base-uncased-MNLI
textattack/roberta-base-MNLI
typeform/distilbert-base-uncased-mnli


sst2
-----------------------------------------------
python code/train.py --model_name "textattack/bert-base-uncased-SST-2" --dataset_name "sst2" --dataset_split "train[:100%]" --total_timesteps 200000 --max_turns 10 --num_envs 4 
Total samples in split: 100
Samples skipped (initial misclassification/error): 8
Samples evaluated: 92
Attack Success Rate (ASR): 35.87% (33/92)
Average Queries per Successful Attack (AQS): 5.58
Average Queries per Evaluated Sample (includes failures): 9.41
Average Queries per Successful Sample (end-to-end): 6.58
Total evaluation time: 5.32 seconds
Saving detailed results to: ./evaluation_results/bert_test_results.json

python code/train.py --model_name "textattack/roberta-base-SST-2" --dataset_name "sst2" --dataset_split "train[:100%]" --total_timesteps 200000 --max_turns 10 --num_envs 4     
--- Evaluation Results ---
Total samples in split: 100
Samples skipped (initial misclassification/error): 5
Samples evaluated: 95
Attack Success Rate (ASR): 42.11% (40/95)
Average Queries per Successful Attack (AQS): 5.00
Average Queries per Evaluated Sample (includes failures): 8.89
Average Queries per Successful Sample (end-to-end): 6.00
Total evaluation time: 5.31 seconds
Saving detailed results to: ./evaluation_results/roberta_test_results.json
Results saved.
--- Evaluation Finished ---

python code/train.py --model_name "textattack/distilbert-base-uncased-SST-2" --dataset_name "sst2" --dataset_split "train[:100%]" --total_timesteps 200000 --max_turns 10 --num_envs 4
Total samples in split: 100
Samples skipped (initial misclassification/error): 44
Samples evaluated: 56
Attack Success Rate (ASR): 37.50% (21/56)
Average Queries per Successful Attack (AQS): 3.71
Average Queries per Evaluated Sample (includes failures): 8.64
Average Queries per Successful Sample (end-to-end): 4.71
Total evaluation time: 3.19 seconds
Saving detailed results to: ./evaluation_results/distilbert_test_results.json


mrpc
-----------------------------------------------------------
python code/train.py --model_name "textattack/bert-base-uncased-MRPC" --dataset_name "mrpc" --dataset_split "train[:100%]" --total_timesteps 200000 --max_turns 10 --num_envs 4
--- Evaluation Results ---
Total samples in split: 100
Samples skipped (initial misclassification/error): 53
Samples evaluated: 47
Attack Success Rate (ASR): 34.04% (16/47)
Average Queries per Successful Attack (AQS): 4.56
Average Queries per Evaluated Sample (includes failures): 9.15
Average Queries per Successful Sample (end-to-end): 5.56
Total evaluation time: 3.76 seconds
Saving detailed results to: ./evaluation_results/bert_mrpc_test_results.json
Results saved.
--- Evaluation Finished ---

PS D:\GitHub Repos\CS6375_Project> python code/train.py --model_name "textattack/roberta-base-MRPC" --dataset_name "mrpc" --dataset_split "train[:100%]" --total_timesteps 200000 --max_turns 10 --num_envs 4    
Total samples in split: 100
Samples skipped (initial misclassification/error): 8
Samples evaluated: 92
Attack Success Rate (ASR): 33.70% (31/92)
Average Queries per Successful Attack (AQS): 5.58
Average Queries per Evaluated Sample (includes failures): 9.51
Average Queries per Successful Sample (end-to-end): 6.58
Total evaluation time: 3.86 seconds
Saving detailed results to: ./evaluation_results/roberta_test_results.json

python code/train.py --model_name "textattack/distilbert-base-uncased-MRPC" --dataset_name "mrpc" --dataset_split "train[:100%]" --total_timesteps 200000 --max_turns 10 --num_envs 4 
Total samples in split: 100
Samples skipped (initial misclassification/error): 11
Samples evaluated: 89
Attack Success Rate (ASR): 7.87% (7/89)
Average Queries per Successful Attack (AQS): 4.71
Average Queries per Evaluated Sample (includes failures): 10.58
Average Queries per Successful Sample (end-to-end): 5.71
Total evaluation time: 4.38 seconds
Saving detailed results to: ./evaluation_results/distilbert_test_results.json

mnli
----------------------------------------------------------------
python code/train.py --model_name "textattack/bert-base-uncased-MNLI" --dataset_name "mnli" --dataset_split "train[:100%]" --total_timesteps 200000 --max_turns 10 --num_envs 4
Total samples in split: 100
Samples skipped (initial misclassification/error): 73
Samples evaluated: 27
Attack Success Rate (ASR): 48.15% (13/27)
Average Queries per Successful Attack (AQS): 3.46
Average Queries per Evaluated Sample (includes failures): 7.85
Average Queries per Successful Sample (end-to-end): 4.46
Total evaluation time: 3.40 seconds
Saving detailed results to: ./evaluation_results/bert_mnli_test_results.json
Results saved.

python code/train.py --model_name "textattack/roberta-base-MNLI" --dataset_name "mnli" --dataset_split "train[:100%]" --total_timesteps 200000 --max_turns 10 --num_envs 4
Total samples in split: 100
Samples skipped (initial misclassification/error): 74
Samples evaluated: 26
Attack Success Rate (ASR): 53.85% (14/26)
Average Queries per Successful Attack (AQS): 3.43
Average Queries per Evaluated Sample (includes failures): 7.46
Average Queries per Successful Sample (end-to-end): 4.43
Total evaluation time: 3.03 seconds
Saving detailed results to: ./evaluation_results/roberta_mnli_test_results.json
Results saved.

python code/train.py --model_name "typeform/distilbert-base-uncased-mnli" --dataset_name "mnli" --dataset_split "train[:100%]" --total_timesteps 200000 --max_turns 10 --num_envs 4
Total samples in split: 100
Samples skipped (initial misclassification/error): 24
Samples evaluated: 76
Attack Success Rate (ASR): 68.42% (52/76)
Average Queries per Successful Attack (AQS): 3.88
Average Queries per Evaluated Sample (includes failures): 6.82
Average Queries per Successful Sample (end-to-end): 4.88
Total evaluation time: 3.26 seconds
Saving detailed results to: ./evaluation_results/roberta_mnli_test_results.json
