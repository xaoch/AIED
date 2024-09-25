import pandas as pd
import chainlit as cl
from pandasai import Agent
from langchain_groq.chat_models import ChatGroq 


@cl.on_chat_start
async def start_chat():
    llm = ChatGroq(model_name="llama3-70b-8192")

     # Sending an image with the local file path
    elements = [
        cl.Image(name="image1", display="inline", path="./robot.jpeg")
        ]
    await cl.Message(content="Hello there, Welcome to Data Analysis Agent!", elements=elements).send()

    files = None

    #### Load external data
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a csv file to begin!", 
            accept=["text/csv"],
            max_size_mb= 100,
            timeout = 180,
        ).send()

    # load the csv data and store in user_session
    file = files[0]

    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    # Read csv file with pandas
    #csv_file = io.BytesIO(file.content)
    df = pd.read_csv(file.path, encoding="utf-8")

    #### Load internal Excel data
    ## Load data
    #df = pd.read_excel('schoolData.xlsx')
    
    ### Load internal CSV data
    #df = pd.read_csv('schoolData.csv')

    dataAgent = Agent(df, 
                      config={"llm": llm},
                      description="You are a data analysis agent. Your main goal is to help non-technical users to analyze data.",
                      )
    # creating user session to store data
    cl.user_session.set('dataAgent', dataAgent)

    # Send response back to user
    # Let the user know that the system is ready
    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()
    


@cl.on_message
async def main(message: cl.Message):
  
    agent = cl.user_session.get('dataAgent')
    
    question = message.content
    response = agent.chat(question)

    msg= None
    if isinstance(response, str) and ".png" in response:
        image = cl.Image(path=response, name="image1", display="inline")
        # Attach the image to the message
        msg = cl.Message(content="Here is the chart:", elements=[image])
    else:
        msg = cl.Message(content=response)
    
    await msg.send()

    # Send final message
    await msg.update()