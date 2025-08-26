from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import create_engine
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent

engine = create_engine("sqlite:///file:Chinook_Sqlite.sqlite?mode=ro&uri=true")
db = SQLDatabase(engine=engine)
llm = ChatOpenAI(model="gpt-5-nano", temperature=0)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type="openai-tools",
)

conversation_history = []

continue_conversation = True
while continue_conversation:
    question = input("Enter a question: ")
    if question.lower() == "exit":
        continue_conversation = False
        break
    
    if conversation_history:
        context = "Previous conversation context:\n"
        for i, (q, a) in enumerate(conversation_history[-3:]):  # Last 3 Q&As
            context += f"Q{i+1}: {q}\nA{i+1}: {a}\n\n"
        full_question = f"{context}Current question: {question}"
    else:
        full_question = question
    
    resp = agent.invoke({"input": full_question})
    answer = resp["output"]
    
    conversation_history.append((question, answer))
    
    print(answer)