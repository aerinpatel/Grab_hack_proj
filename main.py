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
Google_map_api = os.getenv("GOOGLE_MAPS_API_KEY")
Flight_api = os.getenv("FLIGHT_API_KEY")
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

async def geocode_address(address: str) -> dict:
    import aiohttp, os
    print(f"[DEBUG] Geocoding address: {repr(address)}")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("[ERROR] Google API key missing.")

    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={api_key}"

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.json()

    if data.get("status") != "OK":
        raise ValueError(f"Could not geocode address: {address} — API status: {data.get('status')}")

    location = data["results"][0]["geometry"]["location"]
    return {"latitude": location["lat"], "longitude": location["lng"]}


async def call_routes_api(origin: dict, destination: dict, alternatives=False):
    url = "https://routes.googleapis.com/directions/v2:computeRoutes"
    headers = {
        "X-Goog-Api-Key": Google_map_api,
        "Content-Type": "application/json",
        "X-Goog-FieldMask": "routes.distanceMeters,routes.duration,routes.polyline.encodedPolyline,routes.durationWithTraffic"
    }

    body = {
        "origin": {"location": {"latLng": {
            "latitude": origin["latitude"],
            "longitude": origin["longitude"]
        }}},
        "destination": {"location": {"latLng": {
            "latitude": destination["latitude"],
            "longitude": destination["longitude"]
        }}},
        "travelMode": "DRIVE",
        "routingPreference": "TRAFFIC_AWARE_OPTIMAL"
    }

    if alternatives:
        body["computeAlternativeRoutes"] = True

 
    print(f"[DEBUG] Routes API request: {json.dumps(body, indent=2)}")

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=body) as resp:
            result = await resp.json()
            print(f"[DEBUG] Routes API response: {json.dumps(result, indent=2)}")  # Debug response
            return result


def _parse_duration_seconds(duration_str):
    if isinstance(duration_str, str) and duration_str.endswith("s"):
        return int(float(duration_str[:-1]))
    return 0

async def check_traffic(route_info: str) -> str:
    print(f"[DEBUG] check_traffic called with route_info: {repr(route_info)}")

    if not route_info or ";" not in route_info:
        return f"Invalid route info format: {route_info}. Expected 'Origin;Destination'."

    origin_str, destination_str = route_info.split(";", 1)

    try:
        # Use geocode_address to get coordinates for origin and destination
        origin = await geocode_address(origin_str.strip())
        destination = await geocode_address(destination_str.strip())
    except ValueError as e:
        return f"[ERROR] {str(e)}"

    url = (
        "https://maps.googleapis.com/maps/api/directions/json"
        f"?origin={origin['lat']},{origin['lng']}&destination={destination['lat']},{destination['lng']}&key={Google_map_api}"
    )

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.json()

    print(f"[DEBUG] Google Directions API status: {data.get('status')}")

    if data.get("status") != "OK":
        return f"[ERROR] Unable to get directions: {data.get('status')}"

    route = data["routes"][0]
    leg = route["legs"][0]

    distance = leg["distance"]["text"]
    duration = leg["duration"]["text"]

    return (
        f"Traffic between {origin_str} and {destination_str}:\n"
        f"Distance: {distance}, Duration: {duration}"
    )


async def calculate_alternative_route(route_info: str) -> str:
    origin_str, destination_str = route_info.split(";")
    origin_str = origin_str.strip().split(";")[0].strip()
    destination_str = destination_str.strip().split(";")[0].strip()

    if not origin_str or not destination_str:
        return "[ERROR] Origin or destination is missing."

    origin = await geocode_address(origin_str)
    destination = await geocode_address(destination_str)



    data = await call_routes_api(origin, destination, alternatives=True)

    if "routes" not in data or not data["routes"]:
        return "Unable to find an alternative route."

    # Pick alternative route if available
    if len(data["routes"]) > 1:
        best_route = data["routes"][1]
    else:
        best_route = data["routes"][0]

    summary = best_route.get("routeLabel", "Unnamed road")
    duration_with_traffic = best_route.get("durationWithTraffic", best_route.get("duration"))
    distance_meters = best_route.get("distanceMeters", 0)

    return f"Alternative route via {summary}, estimated travel time {duration_with_traffic//60} minutes for {(distance_meters/1000):.1f} km."


async def notify_passenger_and_driver(message: str) -> str:
    
    # Here, we will integrate with Twilio, Firebase Cloud Messaging, or anyother in-app chat API
    print(f"Sending notification: {message}")
    # Simulate push notification delivery
    await asyncio.sleep(1)  
    return "Passenger and driver have been informed."

async def check_flight_status(flight_number: str) -> str:

    url = f"http://api.aviationstack.com/v1/flights?access_key={Flight_api}&flight_iata={flight_number}"

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            data = await resp.json()

    if "data" not in data or not data["data"]:
        return f"Unable to retrieve status for flight {flight_number}."

    flight = data["data"][0]
    status = flight.get("flight_status", "unknown").capitalize()
    departure_time = flight["departure"].get("estimated", "N/A")

    return f"Flight {flight_number} is currently {status}. Estimated departure: {departure_time}."


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
        Tool(
            name="find_nearby_locker",
            func=find_nearby_locker,
            coroutine=find_nearby_locker_async,
            description="Useful for locating a secure parcel locker..."
        ),
        Tool(
            name="check_traffic",
            func=check_traffic,
            coroutine=check_traffic,
            description="Check traffic between two places. Input must be in the format 'Origin;Destination' with no extra text."
        ),        
        Tool(
            name="calculate_alternative_route",
            func=calculate_alternative_route,
            coroutine=calculate_alternative_route,
            description="Calculates an alternative fastest route when obstruction is detected. Input is route details."
        ),
        Tool(
            name="notify_passenger_and_driver",
            func=notify_passenger_and_driver,
            coroutine=notify_passenger_and_driver,
            description="Notifies both passenger and driver with updated route and ETA. Input is the notification message."
        ),
        Tool(
            name="check_flight_status",
            func=check_flight_status,
            coroutine=check_flight_status,
            description="Checks the current status of the passenger's flight. Input is the flight number."
        ),


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


async def classify_scenario(user_input: str) -> str:
    prompt = (
        f"Classify the following situation into exactly one category:\n"
        f"1. GrabExpress - If it involves a delivery partner, recipient, package delivery, drop-off, locker, or recipient not available.\n"
        f"2. GrabCar - If it involves a passenger trip, traffic, airport, urgent travel, rerouting, or flight status.\n"
        f"Return only the exact category name: GrabExpress or GrabCar. no other bs\n\n"
        f"Situation: {user_input}"
    )

    chatHistory = [{"role": "user", "parts": [{"text": prompt}]}]
    payload = {"contents": chatHistory}
    apiKey = Google_api
    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={apiKey}"

    async with aiohttp.ClientSession() as session:
        async with session.post(apiUrl, headers={'Content-Type': 'application/json'}, data=json.dumps(payload)) as response:
            result = await response.json()
            if result.get('candidates') and len(result['candidates']) > 0 and 'parts' in result['candidates'][0]['content']:
                classification = result['candidates'][0]['content']['parts'][0]['text'].strip()
                if classification not in ["GrabExpress", "GrabCar"]:
                    return "GrabExpress"  # default fallback if classification is not recognized
                return classification
            else:
                return "GrabExpress"  # fallback if API fails

async def enhance_grabcar_input(user_input: str) -> str:
    prompt = (
        f"You are a prompt engineer for a GrabCar agent. "
        f"A passenger is on an urgent trip and {user_input}. "
        f"The agent should first always check traffic and if no traffic detected it should exit program otherwise it should generate an actionable goal. "
        f"Your task is to generate an actionable goal for the agent,which first focuses on checking traffic using check_traffic and if traffic or any disruption it should recalculating the route, notifying the passenger, "
        f"and checking the flight status if applicable. Output must be a single, direct sentence."
    )

    chatHistory = [{"role": "user", "parts": [{"text": prompt}]}]
    payload = {"contents": chatHistory}
    apiKey = Google_api
    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={apiKey}"

    async with aiohttp.ClientSession() as session:
        async with session.post(apiUrl, headers={'Content-Type': 'application/json'}, data=json.dumps(payload)) as response:
            result = await response.json()
            if result.get('candidates') and len(result['candidates']) > 0 and 'parts' in result['candidates'][0]['content']:
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                return (
                    "A passenger is on an urgent trip and a major traffic obstruction was detected. "
                    "Recalculate the route, notify the passenger and driver, and check the passenger's flight status."
                )


# main starting point 
async def main():
    delivery_agent = setup_agent()

    global user_scenario_input, recipient_reply
    recipient_reply = ""

    user_input = input("Enter your situation:\n> ")
    user_scenario_input = user_input

    scenario = await classify_scenario(user_input)
    print(f"[DEBUG] Identified scenario: {scenario}")

    if scenario == "GrabExpress":
        agent_goal = await enhance_userinput(user_input)
        await delivery_agent.arun(agent_goal)

    elif scenario == "GrabCar":
        origin = input("Enter the origin (default: Surat):\n> ").strip() or "Surat"
        destination = input("Enter the destination (default: Mumbai):\n> ").strip() or "Mumbai"
        route_info = f"{origin};{destination}"
        
        print(f"[DEBUG] Using route info: {route_info}")
        
        agent_goal = await enhance_grabcar_input(user_input)
        await delivery_agent.arun(agent_goal)


if __name__ == "__main__":
    asyncio.run(main())
