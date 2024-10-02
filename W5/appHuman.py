from crewai import Agent, Task, Crew, Process
from langchain_groq.chat_models import ChatGroq 
import chainlit as cl
from chainlit import run_sync
from crewai_tools import tool

llm = ChatGroq(model_name="groq/llama3-70b-8192")

@tool("Ask Human the question")
def ask_human(question: str) -> str:
    """Ask the question to the human user"""
    human_response  = run_sync( cl.AskUserMessage(content=f"{question}",author="Teacher").send())
    cl.Message(content=human_response).send()
    if human_response:
        return human_response["output"]


@cl.on_chat_start
async def on_chat_start():
    
    # Create a classroom with a teacher and 5 students
    questioner = Agent(
        role='Questioner',
        goal='You ask questions about {topic} to the best of your abilities',
        backstory="""You are an experienced teacher specializing in {topic}.
        You like to create challenging but fair questions about your speciality.
        """,
        verbose=True,
        allow_delegation=False,
        tools=[],
        llm=llm
    )

    grader = Agent(
        role='Grader',
        goal='You like to grade and provide feedback to the answer of students, especially for {topic}',
        backstory="""You are a strict but fair grader for {topic}.  You like to provide extensive feedback on why an answer is correct or wrong and what source material
        students can visit to learn more about it
        """,
        verbose=True,
        allow_delegation=False,
        tools=[],  # No specific tools for now
        llm=llm
    )

    question_generation = Task(
        description="Generate a question about {topic}",
        expected_output="A question redeable by a human student",
        agent=questioner,
    )

    asking_question = Task(
        description="Ask the question to the human user",
        expected_output="The answer from the human",
        agent=questioner,
        tools=[ask_human]
    )

    question_grading = Task(
        description="Evaluate the answer given in a fair way and provide feedback on how to improve",
        expected_output="The grade between 0 and 10, the feedback on what was right or wrong and a list of resources to learn more about it",
        agent=grader
    )


    crew = Crew(
        agents=[questioner,grader],
        tasks=[question_generation,asking_question, question_grading],
        verbose=True,
        process= Process.sequential,
    )    

    cl.user_session.set('crew', crew)

    await cl.Message(content=f"Welcome to the the test preparer.  What topic you want your question be about?", author="Crew").send()

@cl.on_message
async def main(message: cl.Message):
  
    crew = cl.user_session.get('crew')
    
    question = message.content
    inputs = {'topic': question}
    crew_output = crew.kickoff(inputs=inputs)
   
    msg = cl.Message(content=crew_output.raw,author="Evaluator")
    await msg.send()
    # Send final message
    await msg.update()