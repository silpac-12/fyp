from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Load your key securely
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_chatgpt_feedback(prompt, role="You are a helpful medical assistant. Explain results clearly for clinicians."):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": role},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error contacting ChatGPT API: {e}"
