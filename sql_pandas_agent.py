# sql_to_pandas_agent.py
# pip install langchain langchain-community langchain-openai pandas matplotlib sqlalchemy python-dotenv
# For SQLite read-only URI support: SQLAlchemy 2.x
# Make sure OPENAI_API_KEY is set (or load from .env)

import os, re, pathlib
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

load_dotenv()

# ---------- CONFIG ----------
# Open SQLite in READ-ONLY mode (so the agent cannot mutate it)
engine = create_engine("sqlite:///file:Chinook_Sqlite.sqlite?mode=ro&uri=true")
db = SQLDatabase(engine=engine)

llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)

sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
sql_agent = create_sql_agent(
    llm=llm,
    toolkit=sql_toolkit,
    verbose=True,
    agent_type="openai-tools",
    max_iterations=15,
)

# tiny guard to block mutating SQL even though DB is read-only
_MUT = re.compile(r"\b(ALTER|DROP|TRUNCATE|INSERT|UPDATE|DELETE|REPLACE)\b", re.I)

def extract_sql(text: str) -> str | None:
    m = re.search(r"```sql\s*(.*?)```", text, re.S | re.I)
    if m: return m.group(1).strip()
    m = re.search(r"\bSELECT\b.*?;", text, re.S | re.I)
    return m.group(0).strip() if m else None

def run_sql(sql: str) -> pd.DataFrame:
    if _MUT.search(sql or ""):
        raise ValueError(f"Blocked unsafe SQL: {sql}")
    with engine.connect() as conn:
        return pd.read_sql_query(text(sql), conn)

# ---------- DEMO QUERY ----------
user_question = (
    "Please get the data needed to plot a line chart showing the monthly revenue for 2010"
    "Then plot it as a line chart."
)

# 1) Ask the SQL agent for a query (we nudge it to include the SQL)
resp = sql_agent.invoke({
    "input": user_question + "\n\n"
             "Please include ONLY the SQL you used inside a ```sql fenced block. "
             "Limit to ~200 rows."
})
sql_text = resp.get("output") or ""
sql = extract_sql(sql_text)
if not sql:
    raise RuntimeError("SQL agent did not return a usable SQL statement.")
print("\n--- SQL ---\n", sql)

# 2) Execute to a DataFrame
df = run_sql(sql)
print("\nDataFrame shape:", df.shape)
print(df.head())

# ========== OPTION A: SAFE (no code exec) – WE do plotting ==========
# (Keep this as a fallback even if you use the Pandas agent below)
# out_dir = pathlib.Path("report/figs"); out_dir.mkdir(parents=True, exist_ok=True)
# plt.figure()
# x, y = df.columns[:2]  # assume first col is label, second is value
# plt.bar(df[x].astype(str), df[y])
# plt.xticks(rotation=60, ha="right")
# plt.title("Top customers by total spend")
# plt.tight_layout()
# safe_plot_path = out_dir / "customers_top_spend_safe.png"
# plt.savefig(safe_plot_path, bbox_inches="tight")
# plt.close()
# print("Saved safe plot to:", safe_plot_path)

# ========== OPTION B: PANDAS AGENT (code exec) – it does the plotting ==========
# WARNING: This enables LLM-generated Python execution.
eda_agent = create_pandas_dataframe_agent(
    llm,
    df,
    allow_dangerous_code=True,      # enable code exec (keep in Docker sandbox)
    verbose=True,
    max_iterations=15,              # give it a little room
    handle_parsing_errors=True,     # don’t crash on minor formatting
)

plot_cmd = (
    "Return ONLY executable Python (no backticks, no comments, no prose). "
    "Assume `df` is already defined and imported with pandas as pd; "
    "matplotlib.pyplot is imported as plt. "
    "Task: Plot a line chart with x=the month column and y=the total_spent column. "
    "Set a title. Save to 'report/figs/monthly_revenue_agent.png' and close the figure."
)
eda_out = eda_agent.invoke({"input": plot_cmd})
print("\n--- Pandas agent output ---\n", eda_out.get("output"))

print("\nChart paths:")
# print("  Safe (no code exec):", safe_plot_path)
print("  Pandas agent plot  : report/figs/monthly_revenue_agent.png")