# Adversarial Attacks on LLMs with Reinforcement Learning

This project explores the use of reinforcement learning (RL) to generate adversarial attacks on large language models (LLMs). The goal is to train an RL agent to modify input text in a way that causes the target LLM to misclassify it, while adhering to constraints such as semantic similarity and limited modifications.

## Project Structure

### Directory Overview
- **`src/`**: Contains the source code for the project.
  - **`train.py`**: Script to train the RL agent using the PPO algorithm.
  - **`evaluate.py`**: Script to evaluate the trained RL agent on a test dataset.
  - **`llm_interface.py`**: Interface for interacting with Hugging Face LLMs for classification tasks.
  - **`text_actions.py`**: Defines discrete text modification actions for adversarial attacks.
  - **`adversarial_env.py`**: Custom Gymnasium environment for adversarial attack simulation.
- **`.gitignore`**: Specifies files and directories to ignore in version control.
- **`.gitattributes`**: Configures Git's handling of text files.
- **`requirements.txt`**: Lists the Python dependencies required for the project.

### Key Components
1. **Reinforcement Learning Agent**:
   - The RL agent is trained using the Proximal Policy Optimization (PPO) algorithm from the `stable-baselines3` library.
   - The agent interacts with a custom environment (`AdversarialEnv`) to learn how to modify text for successful attacks.

2. **Custom Environment**:
   - The `AdversarialEnv` simulates the adversarial attack process.
   - It provides observations (e.g., probabilities of LLM predictions) and rewards based on the success of the attack and the quality of modifications.

3. **Text Modification Actions**:
   - Actions include replacing words with synonyms, deleting words, adding noise words, swapping words, and more.
   - These actions are defined in `text_actions.py` and are designed to maintain semantic similarity while altering the text.

4. **LLM Interface**:
   - The `LLMInterface` provides a wrapper around Hugging Face models for classification tasks.
   - It handles tokenization, model inference, and label mapping.

5. **Dataset Handling**:
   - The project uses datasets from the Hugging Face `datasets` library, including GLUE tasks like SST-2, MRPC, and MNLI.
   - Preprocessing ensures compatibility with the LLM and RL environment.

## Installation

To set up the project, first install the required dependencies listed in `requirements.txt`. You can do this using the following command:

```bash
pip install -r requirements.txt
```

## Running the Scripts

### Training the RL Agent

To train the RL agent, use the `train.py` script. Below is an example command:

```bash
python src/train.py \
    --model_name textattack/bert-base-uncased-imdb \
    --dataset_name imdb \
    --dataset_split train[:5%] \
    --max_turns 15 \
    --num_envs 4 \
    --total_timesteps 200000 \
    --learning_rate 3e-4 \
    --n_steps 2048 \
    --batch_size 64 \
    --n_epochs 10 \
    --gamma 0.99 \
    --gae_lambda 0.95 \
    --clip_range 0.2 \
    --seed 42 \
    --device auto \
    --log_dir ./tensorboard_logs/ \
    --save_dir ./trained_models/
```

### Evaluating the RL Agent

To evaluate a trained RL agent, use the `evaluate.py` script. Below is an example command:

```bash
python src/evaluate.py \
    --model_path ./trained_models/ppo_shaped_bert-base-uncased-imdb_imdb/final_model.zip \
    --llm_model_name textattack/bert-base-uncased-imdb \
    --dataset_name imdb \
    --dataset_split test[:5%] \
    --max_turns 25 \
    --num_samples 100 \
    --output_file ./evaluation_results/results.json \
    --device auto
```

The evaluation script will output metrics such as attack success rate (ASR) and average queries per sample.

