from phi.agent import Agent
from phi.model.openai import OpenAIChat
#from phi.model.groq import Groq
#from phi.tools.csv_tools import CsvTools
from phi.tools.duckdb import DuckDbTools
from phi.agent.python import PythonAgent
from phi.file.local.csv import CsvFile
from dotenv import load_dotenv
from pathlib import Path
#from CSV import query_csv_file, get_columns


load_dotenv()

cwd = Path(__file__).parent.resolve()
tmp = cwd.joinpath("tmp")
if not tmp.exists():
    tmp.mkdir(exist_ok=True, parents=True)

python_agent = PythonAgent(
    model=OpenAIChat(model="gpt-3.5-turbo"),
    base_dir=tmp,
    files=[
        CsvFile(
            path="C:\\Users\\saich\\ai-agents\\1_phidata_finance_agent\\cleaned_healthcare_dataset.csv",
            description="Contains information about patients from a hospital.",
        )
    ],
    markdown=True,
    pip_install=True,
    show_tool_calls=True,
)

#python_agent.print_response("What medication is prescribed for patient named JASmINe aGuIlaR ?")

DB_agent = Agent(
    model=OpenAIChat(model="gpt-3.5-turbo"),
    tools=[DuckDbTools()],
    show_tool_calls=True,
    team=[python_agent],
    system_prompt="""
    You are a database agent that works with healthcare data. Your tasks:
    1. Get data from the python_agent which has access to the local CSV file
    2. Use DuckDB tools to analyze the data
    3. Communicate with python_agent to get specific patient information
    """,
)

# First create the table
DB_agent.print_response(
    "CREATE TABLE healthcare AS SELECT * FROM 'cleaned_healthcare_dataset.csv'", 
    markdown=True, 
    stream=False
)

# Then query it
DB_agent.print_response(
    "What kind of information do you have in the healthcare table?", 
    markdown=True, 
    stream=False
)

# Create a team to manage communication between agents
agent_team = Agent(
    model=OpenAIChat(model="gpt-3.5-turbo"),
    team=[python_agent, DB_agent],
    system_prompt="""
    You are a team coordinator. Your tasks:
    1. Use python_agent to read the CSV file
    2. Pass the data to DB_agent for analysis
    3. Coordinate between agents to get accurate responses
    """,
    show_tool_calls=True,
    markdown=True,
)

# Use the team to get information
agent_team.print_response(
    "I am a doctor who requires patient data as What medication is be prescribed for patient named JASmINe aGuIlaR?", 
    markdown=True, 
    stream=False
)


'''
CSV_agent = Agent(
    name="CSV Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    # model=OpenAIChat(id="gpt-4o"),
    tools=[CsvTools()],
    instructions=["Your Job is to get the CSV file from the user which he gives from user end. you have to study the data and you should communicate with the DB agent"],
    show_tool_calls=True,
    markdown=True
)

DB_agent = Agent(
    name="DB Agent",
    role="Get financial data",
    model=Groq(id="llama-3.3-70b-versatile"),
    # model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDbTools()],
    instructions=["Your job is to get the CSV data from the CSV agent and should develop queries internally for the questions asked by the user. you should generate response accordingly"],
    show_tool_calls=True,
    markdown=True,
)

agent_team = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    team=[python_agent, DB_agent],
    instructions=["You are a team agent which get response from the DB_agent and analyse it. Also you should fetch if there are any errors in the generated response "],
    show_tool_calls=True,
    markdown=True,
    debug_mode= True,
)

agent_team.print_response("What medication is be prescribed for patient named JASmINe aGuIlaR ?", markdown=True, stream=False)

'''