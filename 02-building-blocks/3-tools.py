import json
import os

import requests
from openai import OpenAI
from pydantic import BaseModel, Field
import random

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

"""
docs: https://platform.openai.com/docs/guides/function-calling
"""

# --------------------------------------------------------------
# Define the tool (function) that we want to call
# --------------------------------------------------------------


def get_weather(latitude, longitude):
    """This is a publically available API that returns the weather for a given location."""
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    )
    data = response.json()
    return data["current"]


def get_random_number() -> dict:
    return {"random_number": random.randint(1, 100)}


# --------------------------------------------------------------
# Step 1: Call model with get_weather tool and get_random_number defined
# --------------------------------------------------------------

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current temperature for provided coordinates in celsius.",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"},
                },
                "required": ["latitude", "longitude"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_random_number",
            "description": "Returns a random number between 1 and 100.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
]

messages = [
    {
        "role": "developer",
        "content": [
            {
                "type": "text",
                "text": "You are a helpful assistant.",
            },
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Generate a random number for me.  What's the weather like in Austin, TX today?",
            },
        ],
    },
]

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    tool_choice="auto",
)

# --------------------------------------------------------------
# Step 2: Model decides to call function(s)
# --------------------------------------------------------------

llm_dump = json.loads(
    json.dumps(completion.model_dump())
)  # load the response from the model

# Extract tool_calls from the response
tool_calls = llm_dump["choices"][0]["message"]["tool_calls"]

# Iterate over tool calls and print function name and arguments
for tool in tool_calls:
    function_name = tool["function"]["name"]
    arguments = json.loads(tool["function"]["arguments"])  # Convert string to dict

    print("-" * 40)
    print(f"Function Name: {function_name}")
    print(f"Arguments: {arguments}")
    print("-" * 40)

# --------------------------------------------------------------
# Step 3: Execute the functions that the model determined need to be executed.
# --------------------------------------------------------------


def call_function(name, args):
    if name == "get_weather":
        return get_weather(**args)
    if name == "get_random_number":
        return get_random_number(**args)


# Handle multiple tool calls
if completion.choices[0].message.tool_calls:
    tool_responses = []
    messages.append(
        completion.choices[0].message
    )  # Add the assistant's message that triggered tool calls

    for tool_call in completion.choices[0].message.tool_calls:
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        # Call the appropriate function
        result = call_function(name, args)

        # Append the response with the correct tool_call_id
        tool_responses.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            }
        )

    # Append tool responses to messages
    messages.extend(tool_responses)

# --------------------------------------------------------------
# Step 4: Supply result and call model again
# --------------------------------------------------------------


class ParsedResponse(BaseModel):
    temperature: float = Field(
        description="The current temperature in celsius for the given location."
    )
    response: str = Field(
        description="A natural language response to the user's question."
    )
    random_number: str = Field(description="A random number")


print(messages)

completion_2 = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    response_format=ParsedResponse,
)

# --------------------------------------------------------------
# Step 5: Check model response
# --------------------------------------------------------------

final_response = completion_2.choices[0].message.parsed
print(f"Temp:  {final_response.temperature}")
print(f"Response: {final_response.response}")
print(f"Random Number: {final_response.random_number}")
