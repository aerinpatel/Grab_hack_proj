<h1 align="center" id="title">Project Synapse</h1>


<p id="description">Developed an asynchronous event-driven Python application that leverages LangChain Google Generative AI (Gemini) and Google Maps Platform to automate decision-making in two operational domains: GrabExpress: Handles delivery partner–recipient communication suggests safe drop-off locations finds secure lockers and processes recipient responses with LLM-based intent parsing. GrabCar: Monitors traffic between origin–destination calculates alternative optimal routes notifies passengers/drivers and checks live flight statuses for urgent trips. Key Features: Natural language classification of user scenarios into logistics or ride-hailing workflows. Prompt engineering for enhanced agent goals and precise LLM responses. Real-time geocoding routing and traffic analysis via Google Maps APIs. Flight status tracking using AviationStack API. Automated timeout handling for unresponsive users. Integrated with asynchronous coroutines for scalable concurrent API calls.</p>

<h2>🚀 Demo</h2>

Explanations :
[https://youtu.be/XnvdVjgUIKA](https://youtu.be/XnvdVjgUIKA)

Code Architecture of Prototype :
[https://app.eraser.io/workspace/72CRzJvRh0oKGkwe4omY?origin=share](https://app.eraser.io/workspace/72CRzJvRh0oKGkwe4omY?origin=share)
  
  
<h2>🧐 Features</h2>

Here're some of the project's best features:

*   AI-Powered Scenario Classification – Automatically classifies situations into GrabExpress (delivery) or GrabCar (ride-hailing) using Google Generative AI (Gemini).
*   LLM-Enhanced Decision Making – Uses prompt-engineered Gemini responses to create actionable goals for agents.
*   Real-Time Map Integration – Integrates Google Maps API for geocoding routing alternative path calculations and traffic-aware trip planning.
*   Recipient Interaction Automation – Initiates chat with recipients suggests safe drop-off locations and locates nearby parcel lockers.
*   Flight Status Tracking – Fetches real-time flight status from AviationStack API for urgent travel scenarios.
*   Timeout Handling – Automatically proceeds with fallback actions if the recipient/passenger is unresponsive.
*   Asynchronous & Scalable – Built with asyncio and aiohttp for concurrent non-blocking API calls.
*   Customizable Agent Tools – Modular tool setup with LangChain making it easy to extend with new APIs or logic.

<h2>🛠️ Installation Steps:</h2>

<p>1. Clone the repository</p>

```
git clone https://github.com/aerinpatel/Grab_hack_proj.git
```

<p>2. Create a virtual environment</p>

```
python -m venv venv
```

<p>3. Activate the virtual environment</p>

```
venv\Scripts\activate
```

<p>4. Install dependencies</p>

```
pip install -r requirements.txt
```

<p>5. Set up environment variables</p>

```
GOOGLE_API_KEY=your_gemini_api_key
GOOGLE_MAPS_API_KEY=your_google_maps_api_key
FLIGHT_API_KEY=your_aviationstack_api_key
```

<p>6. Run the project</p>

```
python main.py
```

  
  
<h2>💻 Built with</h2>

Technologies used in the project:

*   Python
*   LangChain
*   Google Generative AI (Gemini)
*   Google Maps API
*   AviationStack API
*   Aiohttp
*   Asyncio
*   Threading
*   Dotenv
