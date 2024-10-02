import os
from crewai import Agent, Task, Crew, Process
from langchain_groq.chat_models import ChatGroq 
import chainlit as cl
from chainlit import run_sync
from crewai_tools import tool

llm = ChatGroq(model_name="groq/llama3-70b-8192")

@tool("Ask Human follow up questions")
def ask_human(question: str) -> str:
    """Ask human follow up questions"""
    human_response  = run_sync( cl.AskUserMessage(content=f"{question}").send())
    if human_response:
        return human_response["output"]


@cl.on_chat_start
async def on_chat_start():
    
    # Create a classroom with a teacher and 5 students
    teacher = Agent(
        role='Teacher',
        goal='Teach {topic} to the best of your abilities',
        backstory="""You are an experienced teacher specializing in {topic}.
        Your goal is to impart knowledge to a class with varying levels of understanding.
        You enjoy challenges and strive to make complex concepts accessible. You answer students' questions and provide feedback on their understanding.""",
        verbose=True,
        allow_delegation=False,
        tools=[],
        llm=llm
    )

    instructional_designer = Agent(
        role='InstructionalDesigner',
        goal='Create lesson plans in {topic}',
        backstory="""You are a instructional designer that is able to create lessons plans for {topic} following the latest pedagogical approaches.
        You pride yourself of using active learning methods in your designs.
        """,
        verbose=True,
        allow_delegation=False,
        tools=[],  # No specific tools for now
        llm=llm
    )

    evaluator = Agent(
        role='Evaluator',
        goal='To critizice lesson plans to make them better.',
        backstory="""You are very knowledgeable in the latest leearning theories and you like to analyze lessons plans to see if they fit those theories 
        and recommend improvements.""",
        verbose=True,
        allow_delegation=False,
        tools=[],
        llm=llm
    )

    content_creation = Task(
        description="Generate a raw list of the content that need to be covered to learn {topic}",
        expected_output="A list between 5 and 10 content elements",
        agent=teacher,
    )

    lesson_plan_creation = Task(
        description="Create a lession plan for {topic} using the list of content elements suggested by the teacher",
        expected_output="A lesson plan",
        agent=instructional_designer
    )

    lesson_plan_evaluation = Task(
        description="Evaluate the activities in the lesson plan and recommend improvements to make it more active and engaging for students",
        expected_output="A list of improvement points",
        agent=evaluator
    )

    lesson_plan_improvement = Task(
        description="Using the list of recommendations from the evaluator and the original lesson plan, create an improved lesson plan that include those recommendations.",
        expected_output="An improved lesson plan",
        agent=instructional_designer
    )

    class_script_creation = Task(
        description="Convert the improved lesson plan provided by the instructional_designer into a script to execute in the class.",
        expected_output="A class script",
        agent=teacher
    )

    crew = Crew(
        agents=[teacher, instructional_designer, evaluator],
        tasks=[content_creation, lesson_plan_creation, lesson_plan_evaluation, lesson_plan_improvement, class_script_creation],
        verbose=True,
        process= Process.sequential,
    )    

    cl.user_session.set('crew', crew)

    await cl.Message(content=f"Welcome to the class script creator.  Mention a topic you are interested in.", author="Crew").send()

@cl.on_message
async def main(message: cl.Message):
  
    crew = cl.user_session.get('crew')
    
    question = message.content
    inputs = {'topic': question}
    crew_output = crew.kickoff(inputs=inputs)
   
    for output in crew_output.tasks_output:
        msg = cl.Message(content=output.raw,author=output.agent)
        await msg.send()
        # Send final message
        await msg.update()