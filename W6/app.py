from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.prompts.image import ImagePromptTemplate
import base64

import chainlit as cl

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
async def on_chat_start():
    model = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="not-needed", streaming=True)
    with_message_history = RunnableWithMessageHistory(model, get_session_history)
    cl.user_session.set("runnable", with_message_history)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable

    image="No image"
    imageFile=None
    if len(message.elements)>0:
        imageFile=message.elements[0]
    
    msg = cl.Message(content="")

    
    if imageFile is not None:
        image=encode_image(imageFile.path)
    
    response = runnable.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text":message.content},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image}"
                        },
                    },
                ]
            )
        ],
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()], session_id="session1")
    )

    msg=cl.Message(content=response.content)
    await msg.send()
