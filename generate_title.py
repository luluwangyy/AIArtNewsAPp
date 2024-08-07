#generate the title


import replicate
import os
import sys
import json

# Set your API token



def generate_title(description):
    for event in replicate.stream(
        "mistralai/mixtral-8x7b-instruct-v0.1",
        input={
            "top_k": 50,
            "top_p": 0.9,
            "prompt": f"Only write one eye-catching title of a news article about a new art show around an artwork that has this description {description} ",
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

   
    generate_title(description)
    