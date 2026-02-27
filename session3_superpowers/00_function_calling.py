"""
Session 3: Introduction to Function Calling
=============================================
Before we add tools to our voice agent, let's understand the magic
behind it — Function Calling.

Normally, an LLM can only generate text. It can't check the weather,
search the web, or save data. But with function calling, we tell the
LLM: "Hey, here are some tools you CAN use." Then the LLM decides
WHEN to call them and with WHAT arguments.

The flow:
  1. We define functions and describe them to the LLM
  2. User asks a question
  3. LLM decides: "I need to call a function to answer this"
  4. LLM returns a function call request (name + arguments)
  5. WE execute the function and send the result back
  6. LLM uses the result to generate a final answer

Usage:
  python session3_superpowers/00_function_calling.py
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(Path(__file__).parent.parent / ".env")

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


# ============================================================
# Step 1: Define a simple function (our "tool")
# ============================================================

# This is a normal Python function. The LLM doesn't run it —
# WE do. The LLM just decides when to call it.

def get_weather(city: str) -> str:
    """Simulate getting weather for a city."""
    # In real life, this would call a weather API.
    # For now, we return fake data to show the concept.
    fake_weather = {
        "khartoum": "42°C, sunny and hot",
        "cairo": "35°C, sunny",
        "london": "15°C, cloudy with rain",
        "tokyo": "22°C, partly cloudy",
    }
    result = fake_weather.get(city.lower(), f"25°C, nice weather in {city}")
    print(f"  >> get_weather('{city}') was called! Returning: {result}")
    return result


def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    try:
        result = str(eval(expression))
        print(f"  >> calculate('{expression}') was called! Result: {result}")
        return result
    except Exception:
        return "Error: could not calculate"


# ============================================================
# Step 2: Describe the functions for the LLM
# ============================================================

# We need to tell the LLM what functions exist, what they do,
# and what arguments they take. This is the "tool declaration".

weather_tool = types.FunctionDeclaration(
    name="get_weather",
    description="Get the current weather for a city. Use this when the user asks about weather.",
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "city": types.Schema(
                type="STRING",
                description="The city name to get weather for",
            ),
        },
        required=["city"],
    ),
)

calculate_tool = types.FunctionDeclaration(
    name="calculate",
    description="Calculate a math expression. Use this when the user asks to do math.",
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "expression": types.Schema(
                type="STRING",
                description="The math expression to evaluate, e.g. '15 * 37 + 2'",
            ),
        },
        required=["expression"],
    ),
)

# Bundle the tools together
tools = types.Tool(function_declarations=[weather_tool, calculate_tool])


# ============================================================
# Step 3: Ask a question that needs a tool
# ============================================================

print("=" * 50)
print("Function Calling — Step by Step")
print("=" * 50)

question = "What's the weather like in Khartoum?"
print(f"\nUser: {question}")
print("-" * 40)

# Send the question WITH the tool definitions
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=question,
    config=types.GenerateContentConfig(tools=[tools]),
)

# ============================================================
# Step 4: Check if the LLM wants to call a function
# ============================================================

# The LLM doesn't answer directly — it returns a function call!
part = response.candidates[0].content.parts[0]

if part.function_call:
    fn_call = part.function_call
    print(f"\nLLM decided to call: {fn_call.name}()")
    print(f"With arguments: {dict(fn_call.args)}")

    # ========================================================
    # Step 5: Execute the function ourselves
    # ========================================================
    # The LLM doesn't run the function — WE do!
    our_functions = {
        "get_weather": get_weather,
        "calculate": calculate,
    }

    fn = our_functions[fn_call.name]
    result = fn(**dict(fn_call.args))

    # ========================================================
    # Step 6: Send the result back to the LLM
    # ========================================================
    # Now we give the function result to the LLM so it can
    # generate a nice human-readable answer.

    print(f"\n>> Sending result back to LLM...")
    print("-" * 40)

    followup = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Content(role="user", parts=[types.Part.from_text(text=question)]),
            types.Content(role="model", parts=[part]),
            types.Content(
                role="user",
                parts=[types.Part.from_function_response(
                    name=fn_call.name,
                    response={"result": result},
                )],
            ),
        ],
        config=types.GenerateContentConfig(tools=[tools]),
    )

    print(f"\nLLM final answer: {followup.text}")
else:
    # The LLM answered directly (no function needed)
    print(f"\nLLM answered directly: {response.text}")


# ============================================================
# Step 7: Try it with math!
# ============================================================

print("\n" + "=" * 50)
print("Let's try a math question...")
print("=" * 50)

math_question = "What is 1024 * 768?"
print(f"\nUser: {math_question}")
print("-" * 40)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=math_question,
    config=types.GenerateContentConfig(tools=[tools]),
)

part = response.candidates[0].content.parts[0]

if part.function_call:
    fn_call = part.function_call
    print(f"\nLLM decided to call: {fn_call.name}()")
    print(f"With arguments: {dict(fn_call.args)}")

    fn = our_functions[fn_call.name]
    result = fn(**dict(fn_call.args))

    followup = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Content(role="user", parts=[types.Part.from_text(text=math_question)]),
            types.Content(role="model", parts=[part]),
            types.Content(
                role="user",
                parts=[types.Part.from_function_response(
                    name=fn_call.name,
                    response={"result": result},
                )],
            ),
        ],
        config=types.GenerateContentConfig(tools=[tools]),
    )

    print(f"\nLLM final answer: {followup.text}")


# ============================================================
# Step 8: A question that does NOT need a tool
# ============================================================

print("\n" + "=" * 50)
print("Now a question that doesn't need any tool...")
print("=" * 50)

simple_question = "What is function calling in AI?"
print(f"\nUser: {simple_question}")
print("-" * 40)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=simple_question,
    config=types.GenerateContentConfig(tools=[tools]),
)

part = response.candidates[0].content.parts[0]

if part.function_call:
    print(f"\nLLM decided to call: {part.function_call.name}()")
else:
    print(f"\nLLM answered directly (no tool needed): {response.text}")

print("\n" + "=" * 50)
print("KEY TAKEAWAY")
print("=" * 50)
print("""
Function calling lets the LLM USE tools, not just talk.

The LLM does NOT execute functions — it only decides:
  1. WHICH function to call
  2. WHAT arguments to pass

WE execute the function and send the result back.

In LiveKit agents, @function_tool does all of this automatically!
The agent framework handles steps 4-6 for us.
""")
