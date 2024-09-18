from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

import chainlit as cl

store = {}
config = {"configurable": {"session_id": "session1"}}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

@cl.on_chat_start
async def on_chat_start():
    model = ChatGroq(model="mixtral-8x7b-32768", streaming=True)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """"Act as knowledgeable and gentle algebra tutor. Students will ask about how to solve a given equation, you should guide them one step at a time 
            without providing a direct answers, but using the socratic method to help them get closer to the answer. Do not answer everything in just one answer.  Always remember the original equation to provide 
            correct guidance.
            DO NOT GO MORE THAN ONE STEP AT A TIME!
            """,
            ),
            ("human", "{question}"),
        ]
    )
    chain = prompt | model | StrOutputParser()
    with_message_history = RunnableWithMessageHistory(chain, get_session_history)
    cl.user_session.set("runnable", with_message_history)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()], session_id="session1"),
    ):
        await msg.stream_token(chunk)

    await msg.send()

