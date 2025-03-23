import os
from crewai import Crew, Agent, Task
from crewai_tools import BaseTool, ScrapeWebsiteTool
from dotenv import load_dotenv
from typing import List


load_dotenv()

# https://chatgpt.com/c/67e0473c-48e0-800a-9c31-3b3e7a804bb0


# Configuration parameters
INPUT_FILE = "sources.txt"  # Default file name that can be changed
OUTPUT_FILE = "comparison_report.txt"  # Default output file
COMPARISON_TASK_DESCRIPTION = """Receive a list of URLs extracted from the previous task. For each URL, use the 'Scrape Website Tool' to retrieve the content.
Then compare these websites, focusing on long-term memory frameworks for LLM agents.
Write a detailed comparison report."""

# Optional: Set a base directory for the script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCES_FILE = os.path.join(BASE_DIR, INPUT_FILE)

# Ensure sources file exists (for demonstration purposes)
if not os.path.exists(SOURCES_FILE):
    with open(SOURCES_FILE, "w") as f:
        f.write("https://www.example.com\n")
        f.write("https://www.wikipedia.org/wiki/Comparison_of_websites\n")
    print(
        f"Created a sample {SOURCES_FILE}. Please update it with the URLs you want to compare."
    )


class URLExtractorTool(BaseTool):
    name: str = "URL Extractor Tool"
    description: str = "Extracts URLs from a file and returns them as a list."

    def _run(self, file_path: str) -> List[str]:
        if not os.path.isfile(file_path):
            return []
        with open(file_path, "r") as f:
            return [line.strip() for line in f if line.strip()]


# Initialize the tool
website_tool = ScrapeWebsiteTool()
url_extractor_tool = URLExtractorTool()

# AGENTS

url_extractor = Agent(
    role="URL Extractor",
    goal=f"Extract a list of URLs from the {INPUT_FILE} file.",
    backstory="You specialize in file parsing and returning clean data for downstream agents.",
    verbose=True,
    allow_delegation=False,
    tools=[url_extractor_tool],
    llm="gpt-4o",
)

content_analyzer = Agent(
    role="Content Analyst",
    goal="Compare the content of multiple websites based on the provided URLs and write a comprehensive comparison.",
    backstory="You are a highly skilled content analyst with years of experience in identifying similarities and differences between online resources.",
    verbose=True,
    allow_delegation=False,
    tools=[website_tool],  # Pass the instance of the tool class
    llm="gpt-4o",
)

# TASKS

extract_urls_task = Task(
    description=f"Use the tool '{url_extractor_tool.name}' to read the file '{os.path.basename(SOURCES_FILE)}' and return a list of URLs contained inside.",
    expected_output="A list of URLs extracted from the file.",
    agent=url_extractor,
    llm="gpt-4o",
)

compare_websites_task = Task(
    description=COMPARISON_TASK_DESCRIPTION,
    expected_output="A detailed comparison of the content from the extracted URLs.",
    agent=content_analyzer,
    context=[extract_urls_task],  # Use output from URL extractor
    llm="gpt-4o",
)

# CREW

website_comparison_crew = Crew(
    agents=[url_extractor, content_analyzer],
    tasks=[extract_urls_task, compare_websites_task],
    verbose=False,
)

if __name__ == "__main__":
    print(f"Starting the website content comparison process using {INPUT_FILE}...")
    comparison_report = website_comparison_crew.kickoff()
    print("\nComparison Report:")
    print(comparison_report)

    # Save the output to a file
    output_path = os.path.join(BASE_DIR, OUTPUT_FILE)
    with open(output_path, "w") as f:
        f.write(comparison_report)
    print(f"\nReport saved to: {output_path}")

    print("\nProcess finished.")
