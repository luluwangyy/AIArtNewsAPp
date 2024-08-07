#generate this label

import replicate
import os
import sys
import json

# Set your API token



def generate_label(description):
    for event in replicate.stream(
        "mistralai/mixtral-8x7b-instruct-v0.1",
        input={
            "top_k": 50,
            "top_p": 0.9,
            "prompt": f"Write a shore and concise label for the work {description} in the following format: Title, Year: 2024, Medium, Dimensions, Collection, Location",
            "temperature": 0.6,
            "system_prompt": "You are a very helpful, respectful and honest assistant.",
            "length_penalty": 1,
            "max_new_tokens": 1024,
            "prompt_template": "<s>[INST] {prompt} [/INST] ",
            "presence_penalty": 0
        },
    ):
        print(str(event), end="")

if __name__ == "__main__":
    description = sys.argv[1]
   

    replicate_api_key = sys.argv[2]

    # Set API key
    os.environ["REPLICATE_API_TOKEN"] = replicate_api_key

    generate_label(description)