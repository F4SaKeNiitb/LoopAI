import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()

# Configure the API key
# Set your API key as an environment variable: export GEMINI_API_KEY='your-api-key'
api_key = os.environ.get('GEMINI_API_KEY')

if not api_key:
    print("Error: GEMINI_API_KEY environment variable not set")
    print("Get your API key from: https://makersuite.google.com/app/apikey")
    exit(1)

genai.configure(api_key=api_key)

# Initialize the model
model = genai.GenerativeModel('gemini-2.5-pro')

# Test 1: Simple text generation
print("Test 1: Simple text generation")
print("-" * 50)
response = model.generate_content("Write a haiku about coding")
print(response.text)
print()

# Test 2: Chat conversation
print("Test 2: Chat conversation")
print("-" * 50)
chat = model.start_chat(history=[])
response = chat.send_message("Hello! What can you help me with?")
print(f"User: Hello! What can you help me with?")
print(f"Gemini: {response.text}")
print()

response = chat.send_message("Tell me a fun fact about Python programming")
print(f"User: Tell me a fun fact about Python programming")
print(f"Gemini: {response.text}")
print()

# Test 3: Streaming response
print("Test 3: Streaming response")
print("-" * 50)
print("User: Count from 1 to 5")
print("Gemini: ", end="")
response = model.generate_content("Count from 1 to 5", stream=True)
for chunk in response:
    print(chunk.text, end="", flush=True)
print("\n")

print("All tests completed successfully!")