from datetime import datetime
import plyer
import schedule
import time
from groq import Groq
from config import GROQ_API_KEY

SCHEDULE = [
    {"start": "9:30 AM",  "end": "10:00 AM", "activity": "Breakfast"},
    {"start": "10:00 AM", "end": "12:20 PM", "activity": "Coding Session 1 - Projects"},
    {"start": "12:30 PM", "end": "1:00 PM",  "activity": "Break - Anime/Shorts (+ pending work if any)"},
    {"start": "1:00 PM",  "end": "2:00 PM",  "activity": "Gaming - Minecraft"},
    {"start": "2:00 PM",  "end": "3:00 PM",  "activity": "Coding Session 2"},
    {"start": "3:00 PM",  "end": "3:30 PM",  "activity": "External Work - Git / AI reading / Social media"},
    {"start": "3:30 PM",  "end": "4:00 PM",  "activity": "Food"},
    {"start": "4:00 PM",  "end": "6:00 PM",  "activity": "Studies / Coding Session 3"},
    {"start": "6:00 PM",  "end": "6:30 PM",  "activity": "Evening Snacks"},
    {"start": "6:30 PM",  "end": "7:30 PM",  "activity": "Studies"},
    {"start": "7:30 PM",  "end": "9:30 PM",  "activity": "Gym"},
    {"start": "9:30 PM",  "end": "10:30 PM", "activity": "Dinner + Post-gym food"},
    {"start": "10:30 PM", "end": "11:30 PM", "activity": "Anime"},
    {"start": "11:30 PM", "end": "12:30 AM", "activity": "Wind down - Sleep before 12:30 AM"},
]
def get_current_block():
    now = datetime.now().time()
    for block in SCHEDULE:
        start_time = datetime.strptime(block["start"], "%I:%M %p").time()
        end_time = datetime.strptime(block["end"], "%I:%M %p").time()
        if start_time <= now < end_time:
            return block
    return None
def send_notification(title,message):
    plyer.notification.notify(
        title=title,
        message=message,
        timeout=10
        )
client = Groq(api_key=GROQ_API_KEY)
def ask_groq(scheduled_activity, actual_activity):
    prompt = f"Scheduled activity: {scheduled_activity}\nActual activity: {actual_activity} "
    response=client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": "You are a helpful personal assistant that helps me stay accountable to my schedule."},
                  {"role": "user", "content": prompt}]
    )
    reply=response.choices[0].message.content
    return reply

def check_and_notify():
    currnt_block=get_current_block()
    if currnt_block:
        send_notification("Curernt work",currnt_block["activity"])
        actual_activity=input("What are you currently doing? ")
        feedback=ask_groq(currnt_block["activity"], actual_activity)
        print(f"Groq's feedback: {feedback}")
schedule.every(1).minutes.do(check_and_notify)

while True:
    schedule.run_pending()
    time.sleep(1)
