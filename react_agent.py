from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.agents.factory import create_agent
from langchain.tools import tool
from langchain_tavily import TavilySearch
import datetime
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

search_tool = TavilySearch(search_depth="basic")

@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """ Returns the current date and time in the specified format """

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time


tools = [search_tool, get_system_time]

agent = create_agent(llm, tools=tools)

# agent.invoke("When was SpaceX's last launch and how many days ago was that from this instant")

result = agent.invoke(
    {"messages": [{"role": "user", "content": "When was SpaceX's last launch and how many days ago was that from this instant"}]},
    context={"user_role": "expert"}
)
# print(result['title'])


# final_msg = result["messages"][-1]
# print("FINAL ANSWER:", final_msg.content)


def print_react_trace(result):
    for msg in result["messages"]:
        # Support both dict-style and object-style messages
        role = getattr(msg, "type", None) or getattr(msg, "role", None)

        # 1. Question
        if role in ("human", "user"):
            print("\n=== Question ===")
            print(msg.content)

        # 2. Tool result (Observation)
        elif isinstance(msg, ToolMessage) or role == "tool":
            tool_name = getattr(msg, "name", None) or getattr(msg, "tool", "tool")
            print("\n=== Observation from", tool_name, "===")
            print(msg.content)

        # 3. AI messages (Thought / Action / Final Answer)
        elif isinstance(msg, AIMessage) or role in ("assistant", "ai"):
            tool_calls = getattr(msg, "tool_calls", []) or getattr(msg, "additional_kwargs", {}).get("tool_calls", [])

            if tool_calls:
                print("\n=== Thought ===")
                print("I should call a tool to solve this.")
                for tc in tool_calls:
                    name = tc["name"]
                    args = tc.get("args", {})
                    print("\n=== Action ===")
                    print(f"{name}({args})")
            else:
                # No tool calls → final answer
                print("\n=== Final Answer ===")
                print(msg.content)


# Use it:
print_react_trace(result)


