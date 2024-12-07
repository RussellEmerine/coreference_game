from transformers import pipeline
from tqdm import tqdm
from pathlib import Path

# usage described here: https://huggingface.co/docs/transformers/en/main_classes/text_generation
generator = pipeline("text-generation", model="bigscience/bloom-560m")


def generate(prefix: str, completion_dir: Path, n: int):
    """
    Generate sentence completion data.
    """

    replies = []
    for i in tqdm(range(n)):
        reply = generator(prefix, do_sample=True, max_new_tokens=100)
        with open(completion_dir / f"{i + 1}.txt", "w") as f:
            f.write(reply[0]["generated_text"])

    return replies
