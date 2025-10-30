import google.generativeai as genai
from pydantic import BaseModel
from dotenv import load_dotenv
import asyncio
import os
from typing import List, Optional
from datetime import datetime
import json

# Load environment variables
load_dotenv()

# Configure Gemini API
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file.")
genai.configure(api_key=api_key)

# -------------------------------
# Calendar Event Schema (Pydantic)
# -------------------------------
class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: List[str]
    location: Optional[str] = None
    description: Optional[str] = None

# -------------------------------
# Date Validation Tool
# -------------------------------
def validate_date(date_str: str) -> str:
    """Validate and format a date string to YYYY-MM-DD format"""
    formats = ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%B %d, %Y"]
    for fmt in formats:
        try:
            parsed_date = datetime.strptime(date_str, fmt)
            return parsed_date.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return date_str  # Return as-is if no format matches

# -------------------------------
# Gemini Calendar Event Extractor
# -------------------------------
model = genai.GenerativeModel("gemini-2.0-flash-001")

async def extract_calendar_event(text: str) -> CalendarEvent:
    """Extract structured calendar event data from text using Gemini."""
    prompt = f"""
    You are a specialized assistant that extracts calendar events from text.
    Extract all details about events including:
    - Event name
    - Date (in YYYY-MM-DD format)
    - List of participants
    - Location (if mentioned)
    - Description (if available)

    If multiple events are mentioned, focus on the most prominent one.

    Return the output as valid JSON in this schema:
    {{
        "name": "",
        "date": "",
        "participants": [],
        "location": "",
        "description": ""
    }}

    Text:
    {text}
    """

    try:
        response = model.generate_content(prompt)
        text_output = response.text.strip()

        # Clean and parse JSON
        json_start = text_output.find('{')
        json_end = text_output.rfind('}') + 1
        json_text = text_output[json_start:json_end]
        data = json.loads(json_text)

        # Validate date
        if "date" in data:
            data["date"] = validate_date(data["date"])

        return CalendarEvent(**data)
    except Exception as e:
        print("Parsing error:", e)
        print("Raw model response:", response.text if 'response' in locals() else "No response")
        return None

# -------------------------------
# Main Runner
# -------------------------------
async def main():
    simple_text = "Let's have a team meeting on 2023-05-15 with John, Sarah, and Mike."

    complex_text = """
    Hi team,

    I'm scheduling our quarterly planning session for May 20, 2023 at the main conference room.
    All department heads (Lisa, Mark, Jennifer, and David) should attend. We'll be discussing
    our Q3 objectives and reviewing Q2 performance. Please bring your department reports.

    Also, don't forget about the company picnic on 06/15/2023!
    """

    print("\n--- Simple Calendar Extractor ---")
    event1 = await extract_calendar_event(simple_text)
    if event1:
        print("Extracted Event:", event1)
        print(f"Event Name: {event1.name}")
        print(f"Date: {event1.date}")
        print(f"Participants: {', '.join(event1.participants)}")
        if event1.location:
            print(f"Location: {event1.location}")
        if event1.description:
            print(f"Description: {event1.description}")

    print("\n--- Advanced Calendar Extractor ---")
    event2 = await extract_calendar_event(complex_text)
    if event2:
        print("Extracted Event:", event2)
        print(f"Event Name: {event2.name}")
        print(f"Date: {event2.date}")
        print(f"Participants: {', '.join(event2.participants)}")
        if event2.location:
            print(f"Location: {event2.location}")
        if event2.description:
            print(f"Description: {event2.description}")

if __name__ == "__main__":
    asyncio.run(main())
