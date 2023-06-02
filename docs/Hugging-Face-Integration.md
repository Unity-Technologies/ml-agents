# The Hugging Face Integration

The [Hugging Face Hub 🤗](https://huggingface.co/models?pipeline_tag=reinforcement-learning) is a central place **where anyone can share and download models**.

It allows you to:
- **Host** your trained models.
- **Download** trained models from the community.
- Visualize your agents **playing directly on your browser**.

You can see the list of ml-agents models [here](https://huggingface.co/models?library=ml-agents).

We wrote a **complete tutorial to learn to train your first agent using ML-Agents and publish it to the Hub**:

- A short tutorial where you [teach **Huggy the Dog to fetch the stick** and then play with him directly in your browser](https://huggingface.co/learn/deep-rl-course/unitbonus1/introduction)
- A [more in-depth tutorial](https://huggingface.co/learn/deep-rl-course/unit5/introduction)

## Download a model from the Hub

You can simply download a model from the Hub using `mlagents-load-from-hf`.

You need to define two parameters:

- `--repo-id`: the name of the Hugging Face repo you want to download.
- `--local-dir`: the path to download the model.

For instance, I want to load the model with model-id "ThomasSimonini/MLAgents-Pyramids" and put it in the downloads directory:

```sh
mlagents-load-from-hf --repo-id="ThomasSimonini/MLAgents-Pyramids" --local-dir="./downloads"
```

## Upload a model to the Hub

You can simply upload a model to the Hub using `mlagents-push-to-hf`

You need to define four parameters:

- `--run-id`: the name of the training run id.
- `--local-dir`: where the model was saved
- `--repo-id`: the name of the Hugging Face repo you want to create or update. It’s always <your huggingface username>/<the repo name> If the repo does not exist it will be created automatically
- `--commit-message`: since HF repos are git repositories you need to give a commit message.

For instance, I want to upload my model trained with run-id "SnowballTarget1" to the repo-id: ThomasSimonini/ppo-SnowballTarget:

```sh
  mlagents-push-to-hf --run-id="SnowballTarget1" --local-dir="./results/SnowballTarget1" --repo-id="ThomasSimonini/ppo-SnowballTarget" --commit-message="First Push"
```

## Visualize an agent playing

You can watch your agent playing directly in your browser (if the environment is from the [ML-Agents official environments](Learning-Environment-Examples.md))

- Step 1: Go to https://huggingface.co/unity and select the environment demo.
- Step 2: Find your model_id in the list.
- Step 3: Select your .nn /.onnx file.
- Step 4: Click on Watch the agent play
