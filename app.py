import streamlit as st
import base64
import openai

from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from operator import itemgetter
from PIL import Image

openai.api_key = "sk-xxx"

st.title("Multimodal Chat")

def bind_and_run_llm(payload):
    image = payload["image"]
    prompt = payload["prompt"]
    response = openai.ChatCompletion.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": str(prompt)  # Convert the prompt to string
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=300
    )
    return response.choices[0].message['content']


image_template = "{image}"
image_prompt = PromptTemplate.from_template(image_template)
prompt_template = "{question}"
prompt = PromptTemplate.from_template(prompt_template)

chain = (
    {"image": itemgetter("image"), "prompt": prompt} |
    RunnableLambda(bind_and_run_llm)
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        ("How can I help you?", "system")
    ]

if "uploaded_file" not in st.session_state:
    st.session_state["uploaded_file"] = None

for msg, role in st.session_state.get("messages", []):
    if role == "system":
        st.write(msg)
    else:
        st.image(msg, width=200)

uploaded_file = st.sidebar.file_uploader("Upload an image file", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

prompt_input = st.text_input("Your message")

if prompt_input:
    st.session_state["messages"].append((prompt_input, "user"))

    response = ""
    if uploaded_file is not None:
        data = uploaded_file.getvalue()
        b64 = base64.b64encode(data).decode()

        response = chain.invoke({"question": prompt_input, "image": b64})
    else:
        response = "Please upload an image first"

    st.session_state["messages"].append((response, "system"))
    st.write(response)