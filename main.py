# Import necessary libraries
import os
import asyncio
import json
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_google_genai import GoogleGenerativeAI
import aiohttp
import threading

# Load the API key from the .env file
load_dotenv()

# --- Global variables to hold user and recipient input ---
# This is a simple way to pass the user's input into the tool functions.
user_scenario_input = ""
recipient_reply_input = ""

# --- Define the Tools for the Agent ---
# Each function's docstring is crucial for the LLM to understand its purpose.

async def contact_recipient_via_chat(message: str) -> str:
    """
    Sends an automated chat message to the recipient to get instructions for a valuable package.
    Input should be the full message to send. This tool's response is now dynamic based on user and recipient input.
    """
    global user_scenario_input, recipient_reply_input
    print(f"\n--- TOOL: Contacting recipient with message: '{message}' ---")

    # Always prompt for recipient reply (reset before each call)
    recipient_reply_input = ""
    reply = input_with_timeout(
        "\nEnter recipient's reply (e.g., 'leave it with the concierge', 'find a nearby locker', 'safe drop off') [15s timeout]:\n> ",
        15
    ).strip()
    recipient_reply_input = reply

    # If still no reply, set to default
    if not recipient_reply_input:
        recipient_reply_input = "Recipient is not replying"

    # Parse recipient reply using LLM to map to valid tool response
    parsed_reply = await parse_recipient_reply_llm(recipient_reply_input)
    print(f"\n[DEBUG] Recipient input: '{recipient_reply_input}' | Parsed reply: '{parsed_reply}'")
    return parsed_reply


def suggest_safe_drop_off(drop_off_location: str) -> str:
    """
    Suggests a safe location to leave the package, only after getting permission from the recipient.
    Input is the suggested drop-off location as a string.
    """
    print(f"\n--- TOOL: Suggesting safe drop-off at: '{drop_off_location}' ---")
    return f"Package successfully left at the {drop_off_location} as per recipient's permission."

def find_nearby_locker() -> str:
    """
    Finds the nearest and most convenient secure parcel locker as an alternative delivery point.
    This tool does not require an input.
    """
    print("\n--- TOOL: Finding a nearby parcel locker ---")
    return "Found a secure parcel locker located at 'City Center Plaza', 5 minutes away. Access code and details sent to the recipient."

# --- Agent Setup ---
def setup_agent():
    """Initializes the LLM and the agent with the defined tools."""
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")

    llm = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    tools = [
        Tool(
            name="contact_recipient_via_chat",
            func=contact_recipient_via_chat,
            coroutine=contact_recipient_via_chat,  # Register async version for LangChain
            description="Useful for initiating contact with the recipient when they are not available. Input is a message to send."
        ),
        Tool(
            name="suggest_safe_drop_off",
            func=suggest_safe_drop_off,
            description="Useful for proposing a safe location to leave a package, but ONLY after the recipient has given permission. Input is the drop-off location."
        ),
        Tool(
            name="find_nearby_locker",
            func=find_nearby_locker,
            description="Useful for locating a secure parcel locker as a last resort if a safe drop-off is not possible. This tool does not require an input."
        ),
    ]

    agent = initialize_agent(
        tools, 
        llm, 
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
        verbose=True
    )

    return agent

# --- Function to generate a prompt using the Gemini API ---
async def generate_agent_prompt(user_input: str) -> str:
    """
    Uses the Gemini API to take a user's natural language input and craft a precise,
    actionable prompt for the agent.
    """
    print("\n--- Using Gemini API to engineer the agent's goal prompt ---")
    
    prompt_for_llm = (
        f"You are a prompt engineer for a delivery agent. A delivery partner has a valuable package, "
        f"but the recipient is unavailable. The delivery partner describes the situation as: "
        f"'{user_input}'. "
        f"Your task is to generate a concise, actionable goal for the delivery agent, focusing on the problem "
        f"and a clear next step. The goal should be a single, direct sentence. "
        f"Example output: 'A delivery partner has a package but the recipient will be home in 40 minutes. The delivery partner cannot wait. Find a solution.'"
        f"Generate the goal for the following situation: '{user_input}'"
    )

    chatHistory = []
    chatHistory.append({ "role": "user", "parts": [{ "text": prompt_for_llm }] })
    payload = { "contents": chatHistory }
    
    apiKey = "AIzaSyCqusUUev91FJQcG7KvBXVChGbebeHmk-Q"
    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={apiKey}"

    async with aiohttp.ClientSession() as session:
        async with session.post(apiUrl, headers={'Content-Type': 'application/json'}, data=json.dumps(payload)) as response:
            result = await response.json()
            if result.get('candidates') and len(result['candidates']) > 0 and 'parts' in result['candidates'][0]['content']:
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                print("API call failed, using default prompt.")
                return (
                    f"A delivery partner has arrived with a valuable package, but the recipient is unavailable. "
                    f"The delivery partner's situation is: '{user_input}'. "
                    f"You must first contact the recipient, and based on the response, decide the best course of action. "
                    "The delivery partner cannot wait for long periods."
                )


async def parse_recipient_reply_llm(reply: str) -> str:
    """
    Uses Gemini API to parse recipient reply and map it to one of the three valid tool responses.
    """
    prompt = (
        "You are an assistant for a delivery agent. The recipient replied: "
        f"'{reply}'. "
        "Based on this reply, choose the most appropriate action from these options ONLY:\n"
        "1. Concierge: \"Recipient replied: 'I'm not home, can you leave it with the building concierge?' Permission granted.\"\n"
        "2. Locker: \"Recipient replied: 'Please use a nearby parcel locker.'\"\n"
        "3. Safe drop off: \"Recipient replied: 'You can leave it at a safe drop off location.' Permission granted.\"\n"
        "Return ONLY the exact matching response above. If the reply does not match any, return: \"Recipient is not replying, please return the parcel.\""
    )
    chatHistory = []
    chatHistory.append({ "role": "user", "parts": [{ "text": prompt }] })
    payload = { "contents": chatHistory }
    apiKey = "AIzaSyCqusUUev91FJQcG7KvBXVChGbebeHmk-Q"
    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={apiKey}"

    async with aiohttp.ClientSession() as session:
        async with session.post(apiUrl, headers={'Content-Type': 'application/json'}, data=json.dumps(payload)) as response:
            result = await response.json()
            if result.get('candidates') and len(result['candidates']) > 0 and 'parts' in result['candidates'][0]['content']:
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                return "Recipient is not replying, please return the parcel."

def input_with_timeout(prompt, timeout):
    """
    Synchronous input with timeout. Returns empty string if timeout.
    """
    result = {"reply": ""}
    def get_input():
        result["reply"] = input(prompt)
    thread = threading.Thread(target=get_input)
    thread.daemon = True
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        return ""
    return result["reply"]

# --- Main Execution ---
async def main():
    delivery_agent = setup_agent()
    
    # Get user input for the scenario
    user_input = input("Enter the delivery partner's situation (e.g., 'The recipient is telling me to wait for 40 mins'):\n> ")
    
    # Set the global variable for the dynamic tool response
    global user_scenario_input, recipient_reply_input
    user_scenario_input = user_input
    recipient_reply_input = ""  # Reset for each run
    
    # Use prompt engineering to create a clear and actionable prompt for the agent
    agent_goal = await generate_agent_prompt(user_input)
    print(f"\n--- Generated Agent Goal: {agent_goal} ---")
    # Run the agent with the dynamically created prompt (async version)
    await delivery_agent.arun(agent_goal)

if __name__ == "__main__":
    asyncio.run(main())
    asyncio.run(main())
