import chainlit as cl
import openai
import os
from config import open_ai_key, serp_api_key

os.environ['OPENAI_API_KEY'] = open_ai_key
os.environ['SERPAPI_API_KEY'] = serp_api_key


def get_gpt_output(user_message):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role":"system", "content":"understand user's request carefully, you need to help them with high accuracy, do not hallucinate!"},
            {"role":"user","content": user_message}
        ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    return response

@cl.on_message
async def main(message : str):
    await cl.Message(content = f"{get_gpt_output(message)['choices'][0]['message']['content']}",).send()

