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

    post_writer = Agent(
        role="Instagram post writer",
        goal="People quickly and easily understand complex AI-related information.",
        backstory="AI-related articles are being published incredibly quickly, and people find it hard to keep up with this information. You need to become a portal that makes it easy for people to access this information.",
    )

    research_about_AI = Task(
        description="research the top five most influential AI-related news topics in Korea as of today.",
        expected_output="A comprehensive list of 5 news on the latest AI trends.",
        agent=researcher,
        tools=[search_tool],
    )

    write_instagram_post = Task(
        description="You will receive five AI-related articles today. You need to create Instagram posts in Korean from these articles that are intuitive and witty, making them quick to read and easy to understand for people.",
        expected_output="A long text where each article is separated by line breaks, with titles and content distinctly organized.",
        agent=post_writer,
    )

    crew = Crew(
        agents=[researcher, post_writer],
        tasks=[research_about_AI, write_instagram_post],
        share_crew=True,
    )

    result = crew.kickoff()

    print(result)
