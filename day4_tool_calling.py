# ===============================
# IMPORTS
# ===============================
import json
import os
import re
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

# ===============================
# SET API KEY
# ===============================
os.environ["TAVILY_API_KEY"] = "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"

# ===============================
# LLM (OLLAMA)
# ===============================
llm = ChatOllama(model="llama3.2:3b")

# ===============================
# TOOL 1: WEB SEARCH
# ===============================
search = TavilySearch()

@tool
def web_search(query: str) -> str:
    """Search latest info from web"""
    result = search.run(query)

    # Extract useful text only
    if isinstance(result, dict) and "results" in result:
        texts = [r["content"] for r in result["results"] if "content" in r]
        return " ".join(texts[:3])  # top 3 results

    return str(result)

# ===============================
# TOOL 2: SUMMARIZE
# ===============================
@tool
def summarize(text: str) -> str:
    """Summarize text"""
    prompt = f"Summarize in a short paragraph:\n{text}"
    return llm.invoke(prompt).content

# ===============================
# TOOL 3: NOTES
# ===============================
@tool
def notes(text: str) -> str:
    """Convert into notes"""
    prompt = f"Convert into notes with Title and Content:\n{text}"
    return llm.invoke(prompt).content

# ===============================
# TOOL MAP
# ===============================
tools = {
    "web_search": web_search,
    "summarize": summarize,
    "notes": notes
}

# ===============================
# AGENT LOOP
# ===============================
def agent_loop(query):
    print(f"\nUser: {query}")

    system_prompt = """
    You are an AI agent with tools:

    1. web_search(query: string)
    2. summarize(text: string)
    3. notes(text: string)

    RULES:
    - Use EXACT parameter names:
      web_search → {"query": "..."}
      summarize → {"text": "..."}
      notes → {"text": "..."}
    - NEVER use "param"
    - If user asks for latest info → use web_search
    - Then summarize ONLY ONCE
    - After that → return final_answer

    Respond ONLY in JSON:

    {"tool": "tool_name", "args": {"parameter_name": "value"}}
    OR
    {"final_answer": "your answer"}
    """

    messages = system_prompt + "\nUser: " + query
    last_result = ""

    for i in range(3):

        response = llm.invoke(messages).content
        print("\nLLM:", response)

        try:
            json_text = re.search(r'\{.*\}', response, re.DOTALL).group()
            data = json.loads(json_text)
        except:
            print("\n⚠️ JSON Error → using raw response")
            print("\n✅ Final Answer:", response)
            break

        # FINAL ANSWER
        if "final_answer" in data:
            print("\n✅ Final Answer:", data["final_answer"])
            break

        # TOOL CALL
        tool_name = data.get("tool")
        args = data.get("args", {})

        print(f"\n🔧 Tool Used: {tool_name}")

        # Fix wrong params
        if tool_name == "web_search" and "query" not in args:
            args["query"] = args.get("param", "")

        if tool_name in ["summarize", "notes"] and "text" not in args:
            args["text"] = args.get("param", "")

        # Stop repeated summarize
        if i > 0 and tool_name == "summarize":
            print("\n🛑 Stopping repeated summarize")

            final_prompt = f"Give a final clear answer based on this:\n{last_result}"
            final_answer = llm.invoke(final_prompt).content

            print("\n✅ Final Answer:", final_answer)
            break

        # Execute tool
        result = tools[tool_name].invoke(args)
        last_result = result

        print("\n📤 Result:", result)

        # Feed result back
        messages += f"\nTool result: {result}\nWhat should you do next?"

# ===============================
# TEST CASES
# ===============================
if __name__ == "__main__":

    agent_loop("What is the latest news on OpenAI?")

    agent_loop("Summarize this paragraph: Artificial Intelligence is transforming industries by automating tasks and improving efficiency.")

    agent_loop("Find the latest news on AI agents and summarize it")