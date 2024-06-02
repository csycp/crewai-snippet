from dotenv import load_dotenv
import os
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool

load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

os.environ["SERPER_API_KEY"] = SERPER_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


search_tool = SerperDevTool()


def main():
    researcher = Agent(
        role="AI news researcher",
        goal="Research and understand new information about the AI industry, which is introducing various technologies.",
        backstory=(
            "People are currently confused by the rapid development of AI technology, particularly LLM technology. Some see AI as an opportunity, while others are skeptical, viewing it as a technology that could destroy the world. In order for humans not to be dominated by technology but to use it as a tool, you need to understand and provide AI-related articles and news to help people understand AI."
        ),
        tools=[search_tool],
    )

    research_about_AI = Task(
        description="research the top five most influential AI-related news topics in Korea as of today.",
        expected_output="A comprehensive list of 5 news on the latest AI trends.",
        agent=researcher,
        tools=[search_tool],
    )

    crew = Crew(
        agents=[researcher],
        tasks=[research_about_AI],
    )

    result = crew.kickoff()

    print(result)
