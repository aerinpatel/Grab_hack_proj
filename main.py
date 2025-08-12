#importing necessary libraries :- (we used langchain , google gen ai , iohttp(for prompt enhancing) , dotenv(for env variables))
import os
import asyncio
import json
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_google_genai import GoogleGenerativeAI
import aiohttp
import threading


load_dotenv() 
Google_api = os.getenv("GOOGLE_API_KEY")
user_scenario_input = ""
recipient_reply = ""


async def contact_recipient_via_chat(message: str) -> str:

    global user_scenario_input, recipient_reply
    print(f"{message}")

    recipient_reply = ""
    reply = input_with_timeout(
        "\nEnter your reply for the delivery agent  [150s timeout]:\n> ",
        150
    ).strip()
    recipient_reply = reply

    if not recipient_reply:
        recipient_reply = "Recipient is not replying"

    parsed_reply = await enhancing_reply(recipient_reply)
    return parsed_reply


async def suggest_safe_drop_off(drop_off_location: str) -> str:
    global recipient_reply

    # I am asking for his preference here 
    recipient_reply = input_with_timeout(
        "\nCan you suggest a specific place for the drop-off? [150s timeout]:\n> ",
        150
    ).strip()

    if not recipient_reply:
        print("\nRecipient did not respond. Finding a nearby locker...")
        return find_nearby_locker()

    # Parse the recipient's reply using LLM
    parsed_reply = await parse_drop_off_location(recipient_reply)

    return parsed_reply

async def find_nearby_locker_async(_input=None):
    find_nearby_locker(_input)


def find_nearby_locker(_input=None):
    # we will integrate here google map api by taking delivery boy's current 
    # location as input and providing a safe drop-off location as output
    print("Found a secure parcel locker located at 'City Center Plaza', 5 minutes away from your home address. I am leaving your parcel safely. Access code and details sent to you via chat.")
    exit(0)


def setup_agent():

    if not Google_api:
        raise ValueError("GOOGLE_API_KEY not found in env file") #if we are not able to find our gemini api key 

    llm = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)  # using free teir gemini api key 
    # temperature set to 0 right now to get deterministic output ( for removing randomness for now ), can be changed later if needed
    tools = [
        Tool(
            name="contact_recipient_via_chat",
            func=contact_recipient_via_chat,
            coroutine=contact_recipient_via_chat, 
            description="Useful for initiating contact with the recipient when they are not available. Input is a message to send."
        ),
        Tool(
            name="suggest_safe_drop_off",
            func=suggest_safe_drop_off,
            coroutine=suggest_safe_drop_off,
            description="Useful for proposing a safe location to leave a package, but ONLY after the recipient has given permission. Input is the drop-off location."
        ),
        # Tool(
        #     name="find_nearby_locker",
        #     func=find_nearby_locker,
        #     description="Useful for locating a secure parcel locker as a last resort if a safe drop-off is not possible. This tool does not require an input."
        # ),
        Tool(
            name="find_nearby_locker",
            func=find_nearby_locker,
            coroutine=find_nearby_locker_async,
            description="Useful for locating a secure parcel locker..."
        )

    ]

    agent = initialize_agent(
        tools, 
        llm, 
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  
        verbose=True # Added for thought of chains 
    )

    return agent

async def enhance_userinput(user_input: str) -> str:
    
    
    prompt = (
        f"You are a prompt engineer for a delivery agent. A delivery partner has a valuable package, "
        f"but the recipient is unavailable. The delivery partner describes the situation as: "
        f"'{user_input}'. "
        f"Your task is to generate a actionable goal for the delivery agent, focusing on the problem "
        f"and a clear next step. The goal should be a single, direct sentence. "
        f"Example output: 'A delivery partner has a package but the recipient is not available  Can I leave it with the building concierge?. The delivery partner cannot wait. Find a solution.'"
        f"Generate the goal for the following situation: '{user_input}'"
    )

    chatHistory = []
    chatHistory.append({ "role": "user", "parts": [{ "text": prompt}] })
    payload = { "contents": chatHistory }
    
    apiKey = Google_api #yaha 2.5 use ho rha hai 
    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={apiKey}"

    async with aiohttp.ClientSession() as session:
        async with session.post(apiUrl, headers={'Content-Type': 'application/json'}, data=json.dumps(payload)) as response:
            result = await response.json()
            # print(f"[DEBUG] Gemini API response: {result}")  # Debugging line to check the API response
            if result.get('candidates') and len(result['candidates']) > 0 and 'parts' in result['candidates'][0]['content']:
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                print("API call failed, using default prompt.") # in case free tier gemini api limit is reached or any other issue
                return ( #returns with this default msg
                    f"A delivery partner has arrived with a valuable package, but the recipient is unavailable. "
                    f"The delivery partner's situation is: '{user_input}'. "
                    f"You must first contact the recipient, and based on the response, decide the best course of action. "
                    "The delivery partner cannot wait for long periods."
                )


async def enhancing_reply(reply: str) -> str:
    prompt = (
        "You are an assistant for a delivery agent. The recipient replied: "
        f"'{reply}'. "
        "Based on this reply, choose the most appropriate action from these options ONLY:\n"
        "1. Locker: \"Recipient replied: 'Please use a nearby parcel locker.'\"\n"
        "2. Safe drop off: \"Recipient replied: 'You can leave it at a safe drop off location.' Permission granted.\"\n"
        "Return ONLY the exact matching response above. If the reply does not match any, return:  'Please use a nearby parcel locker.'"
    )

    chatHistory = []
    chatHistory.append({ "role": "user", "parts": [{ "text": prompt }] })
    payload = { "contents": chatHistory }
    apiKey = Google_api
    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={apiKey}" #2.5 


    async with aiohttp.ClientSession() as session:
        async with session.post(apiUrl, headers={'Content-Type': 'application/json'}, data=json.dumps(payload)) as response:
            result = await response.json()
            # print(result)
            if result.get('candidates') and len(result['candidates']) > 0 and 'parts' in result['candidates'][0]['content']:
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                return "Recipient is not replying, find some secure locker "


async def parse_drop_off_location(reply: str) -> str:
    prompt = (
        f"The recipient provided the following location for the parcel drop-off: '{reply}'. "
        f"Please return a single, clear sentence in the format: 'Your parcel has been safely delivered to {reply}, as per your instructions.' "
        "Ensure the location is extracted and formatted properly."
    )

    chatHistory = []
    chatHistory.append({ "role": "user", "parts": [{ "text": prompt }] })
    payload = { "contents": chatHistory }
    apiKey = Google_api
    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={apiKey}"

    async with aiohttp.ClientSession() as session:
        async with session.post(apiUrl, headers={'Content-Type': 'application/json'}, data=json.dumps(payload)) as response:
            result = await response.json()
            if result.get('candidates') and len(result['candidates']) > 0 and 'parts' in result['candidates'][0]['content']:
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                return "Unknown location"


def input_with_timeout(prompt, timeout): #if user is not replying for 150 seconds 
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





# main starting point 
async def main():
    delivery_agent = setup_agent()
    
    
    user_input = input("Hii Delivery Agent send your message to the recipient \n> ")
    
    global user_scenario_input, recipient_reply
    user_scenario_input = user_input
    recipient_reply = "" 
    
    agent_goal = await enhance_userinput(user_input)  # just used for enhancing the dilivery guy provided input 
    
    
    await delivery_agent.arun(agent_goal) #finally run based upon generated agent goal 

if __name__ == "__main__":
    asyncio.run(main())
    # asyncio.run(main())
