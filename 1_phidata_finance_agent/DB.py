from phi.agent import Agent
from phi.tools.duckdb import DuckDbTools

agent = Agent(
    tools=[DuckDbTools()],
    show_tool_calls=True,
    system_prompt="Use this file for Movies data: C:\\Users\\saich\\Favorites\\Mohan\\GenerativeAI\\Python\\PythonPrograms\\Data_Manpulation\\cleaned_healthcare_dataset.csv",
)
agent.print_response("What medication is be prescribed for patient named JASmINe aGuIlaR ?", markdown=True, stream=False)

