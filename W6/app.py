from enum import auto, Enum
import json
import dataclasses
from typing import List
import aiohttp
from PIL import Image
import io
import os
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
import base64
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.prompts.image import ImagePromptTemplate
import base64

import chainlit as cl
from chainlit.input_widget import Select, Slider

store = {}
config = {"configurable": {"session_id": "session1"}}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

def encode_image(image_path):
    ''' Getting the base64 string '''
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


@cl.on_chat_start
async def start():
    settings = await cl.ChatSettings(
        [
            Select(
                id="model",
                label="Model",
                values=["llava-v1.5-13b"],
                initial_index=0,
            ),
            Slider(
                id="temperature",
                label="Temperature",
                initial=0,
                min=0,
                max=1,
                step=0.1,
            ),
            Slider(
                id="top_p",
                label="Top P",
                initial=0.7,
                min=0,
                max=1,
                step=0.1,
            ),
            Slider(
                id="max_token",
                label="Max output tokens",
                initial=512,
                min=0,
                max=1024,
                step=64,
            ),
        ]
    ).send()

    model = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="not-needed", streaming=True)
    with_message_history = RunnableWithMessageHistory(model, get_session_history)
    cl.user_session.set("runnable", with_message_history)


@cl.on_settings_update
async def setup_agent(settings):
    cl.user_session.set("settings", settings)


@cl.on_message
async def main(message: cl.Message):
    image = next(
        (
            file.path
            for file in message.elements or []
            if "image" in file.mime and file.path is not None
        ),
        None,
    )

    runnable = cl.user_session.get("runnable")  # type: Runnable

    msg = cl.Message(content="")

    if image:
        encoded_image=encode_image(image)
        url=f"data:image/jpeg;base64,{encoded_image}"
        response = runnable.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text":message.content},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": url
                        },
                    },
                ]
            )
        ],
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()], session_id="session1")
        )
    else:
        response = runnable.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text":message.content},
                ]
            )
        ],
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()], session_id="session1")
        )
    
    msg=cl.Message(content=response.content)
    await msg.send()