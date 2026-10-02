
# LLaMA Model Setup

## Prerequisites

Before running the LLaMA model, you must log in to your Hugging Face account. Without authentication, you will not have permission to access the model.

### Login to Hugging Face

```bash
huggingface-cli login
```
and then you will prompted to enter your LLMA_TOKEK

Or another way you can do it in you code direction, is by adding this like right above before you load your model:

```bash
LLAMA_TOKEN = os.getenv("LLAMA_TOKEN")

# Paste your token here
login(token=LLAMA_TOKEN)
```
You will be prompted to enter your Hugging Face API token. Get your token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

## Running the Model

Once authenticated, you can run the LLaMA model:

```bash
python run_llama.py
```

## Notes

- Authentication is required before each session
- Ensure you have accepted the LLaMA model license on Hugging Face
- Check your internet connection during model download
