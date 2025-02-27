from openai import OpenAI, RateLimitError
import os
from halo import Halo
import backoff
import time
from pydantic import BaseModel

# setup the OpenAI Client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Azure OpenAI variables from .env file
OPENAI_MODEL = os.getenv("OPENAI_MODEL")


class ModelResponse(BaseModel):
    text: str
    model: str


def open_file(filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as infile:
        return infile.read()


###  OpenAI chat completions call with backoff for rate limits
@backoff.on_exception(backoff.expo, RateLimitError)
def chat(**kwargs):
    try:
        spinner = Halo(text="Reasoning...", spinner="dots")
        spinner.start()
        #print(kwargs)

        start_time = time.time()  # Record the start time
        response = client.beta.chat.completions.parse(**kwargs)
        end_time = time.time()  # Record the end time

        elapsed_time = end_time - start_time  # Calculate the elapsed time in seconds
        minutes, seconds = divmod(
            elapsed_time, 60
        )  # Convert seconds to minutes and seconds
        formatted_time = (
            f"{int(minutes)} minutes and {seconds:.2f} seconds"  # Format the time
        )

        text = response.choices[0].message.parsed.text
        model = response.model
        tokens = response.usage

        spinner.stop()

        return text, model, tokens, formatted_time
    except Exception as yikes:
        print(f'\n\nError communicating with OpenAI: "{yikes}"')
        exit(0)


def main():
    while True:
        # Get user query
        query = input(f"\nMain question (using model: {OPENAI_MODEL}): ")
        if query.lower() == "exit":
            exit(0)

        prompts = [
            "What information do I already know about this topic? What information do I need to recall into my working memory to best answer this?",
            "What techniques or methods do I know that I can use to answer this question or solve this problem? How can I integrate what I already know, and recall more valuable facts, approaches, and techniques?",
            "And finally, with all this in mind, I will now discuss the question or problem and render my final answer.",
        ]
        # Create conversation
        conversation = list()
        # conversation.append({'role': 'system', 'content': open_file('./prompts/system.md')})
        conversation.append(
            {
                "role": "system",
                "content": open_file("./prompts/latent_space.xml").replace(
                    "{{query}}", query
                ),
            }
        )

        for p in prompts:
            # conversation.append({'role': 'user', 'content': p})
            conversation.append(
                {"role": "user", "content": [{"type": "text", "text": p}]}
            )
            print("\n\n\nUSER: %s" % p)
            text, model, tokens, formatted_time = chat(
                model=OPENAI_MODEL,
                messages=conversation,
                max_completion_tokens=2000,
                temperature=1,
                response_format=ModelResponse,
            )
            conversation.append(
                {"role": "assistant", "content": [{"type": "text", "text": text}]}
            )
            print("\n\n\nAssistant:\n%s" % text)


        print(f"Model used: {model}")
        print(f"Your question took a total of: {tokens.total_tokens} tokens")
        print(
            f"Your question took: {tokens.completion_tokens_details.reasoning_tokens} reasoning tokens"
        )
        print(f"Your question prompt used: {tokens.prompt_tokens_details}")
        print(f"Time elapsed: {formatted_time}")


if __name__ == "__main__":
    main()