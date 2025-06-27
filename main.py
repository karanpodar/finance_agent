from crewai import Agent, Task, Crew
from sentiment_agent import main as agent1_main
from technical_agent import analyze_stock as agent2_main


agent1 = Agent(
    role="Sentiment Analyst",
    goal="Run a workflow to analyze sentiment of a stock", 
    backstory="An expert in orchestrating logic and workflows to analyze stock sentiments",
    allow_delegation=False
)

agent2 = Agent(
    role="Technical Analyst",
    goal="Run a workflow to analyze technical indicators of a stock",
    backstory="An expert in analyzing stock technical indicators and patterns",
    allow_delegation=False
)

task1 = Task(
    description="Run the sentiment analysis workflow",
    expected_output="Sentiment analysis results stored in a file",
    agent=agent1,
    function_call=agent1_main
)

task2 = Task(
    description="Run the technical analysis workflow",
    expected_output="Technical analysis results stored in a file",
    agent=agent2,
    function_call=agent2_main
)

crew = Crew(
    agents=[agent1, agent2],
    tasks=[task1, task2],
    verbose=True,
    # process_type="parallel"
    process_type="sequential"
)

if __name__ == "__main__":
    results = crew.kickoff()
    for i, res in enumerate(results, start=1):
        print(f"\n✅ Result from Task {i}:\n{res}")
