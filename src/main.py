from dotenv import load_dotenv
import os
from crewai import Agent, Task, Crew

load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


def main():
    joker = Agent(
        role="joke generator",
        goal="generate very funny joke",
        backstory=(
            "The world is becoming darker and people are losing their laughter. It's time we need some funny jokes for adults to brighten things up again."
        ),
    )

    generate_joke = Task(
        description=(
            "Make up a very very naughty and funny joke in Korean. It should be in the form of a dialogue."
        ),
        expected_output="dialogue-style joke.",
        agent=joker,
    )

    crew = Crew(
        agents=[joker],
        tasks=[generate_joke],
    )

    result = crew.kickoff()

    print(result)
