from groq import Groq
from config import GROQ_API_KEY
client = Groq(api_key=GROQ_API_KEY)
messages = [
    {"role": "system", "content": "You are a helpful personal assistant"}
]
while True:
    user_input=input("You: ")
    if user_input=="exit":
        break
    
    messages.append({"role": "user", "content": user_input}  )

    response = client.chat.completions.create(
      model="llama-3.3-70b-versatile",
        messages=messages,
    )
    reply=response.choices[0].message.content
    print(f"Bot: {reply}")
    messages.append({"role": "assistant", "content": reply})

