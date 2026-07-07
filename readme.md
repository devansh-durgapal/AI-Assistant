# Assistant

A lightweight AI chatbot project built with LangChain, Groq, Google Gemini, Chroma, and DuckDuckGo search. The app routes each user query to either web search or direct model answering, and it also includes an experimental personal-memory branch backed by ChromaDB.

## What it does

- Classifies incoming queries before answering.
- Uses DuckDuckGo search for current or web-based questions.
- Uses Groq / Gemini model chains for direct responses.
- Stores and retrieves conversational memory with ChromaDB.

## Project Structure

- `chatbot.py` - main CLI entrypoint and routing logic.
- `src/LlmModel.py` - Groq and Google model wrappers plus prompts.
- `src/searchEngine.py` - search engine adapters.
- `src/Chroma.py` - ChromaDB persistence and memory lookup.
- `src/branches.py` - experimental memory-based response branch.
- `database/Chroma_db/` - local Chroma persistence directory.

## Requirements

- Python 3.13 or compatible.
- A virtual environment is recommended.
- API keys for the models you want to use.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root and add the keys you need:

   ```env
   GROQ_API_KEY=your_groq_key
   GOOGLE_API_KEY=your_google_key
   TAVILY_API_KEY=your_tavily_key
   ```

   `TAVILY_API_KEY` is only needed if you use the Tavily search adapter.

## Run

Start the chat loop from the project root:

```bash
python chatbot.py
```

Type a message and press Enter. Enter `exit` to quit.

## How It Works

1. `chatbot.py` loads environment variables and builds a routing chain.
2. The Groq classifier decides whether a query should go to search or direct answering.
3. Search queries are sent to DuckDuckGo.
4. Other queries are answered by the model chain.

## Memory Mode

The experimental branch in `src/branches.py` shows how previous user/assistant exchanges can be stored in ChromaDB and reused for similar questions. It uses the local database under `database/Chroma_db/`.

## Notes

- The repository contains exploratory code paths, so not every module is wired into the main entrypoint.
- If you change the database path or model names, update the corresponding files in `src/`.
