# trackSabha 🏛️

An open, FastAPI-powered civic AI that lets you chat about Indian Parliament and explore a live knowledge graph of entities and relationships, rendered with D3.js — all backed by MongoDB and a simple data pipeline.

—

## 🚀 Overview

trackSabha is a lightweight web application that:

- Ingests parliamentary debate transcripts (YouTube) and cleans them for analysis
- Builds a knowledge graph (entities, relationships) from transcripts and Q&A context
- Lets you chat with an AI assistant that provides structured results and follow-up suggestions
- Automatically embeds a D3.js knowledge graph inline with each answer (with a minimize toggle)
- Persists chat sessions across reloads (MongoDB + localStorage), including rendered assistant content

—

## ✨ Key Features

- Chat UX with structured responses
	- Response cards (expand/collapse), follow-up suggestions (single click), rich links
	- Server-rendered HTML is stored with messages for exact rehydration on reload

- Always-on Knowledge Graph
	- Inline knowledge graph card after each assistant response (no separate button)
	- Minimize/expand per graph, unique IDs to avoid collisions, friendly error fragments when empty
	- D3.js visualization (force layout), zoom/pan, node details on click

- Robust Session Persistence
	- Session ID preserved across “Clear Chat”
	- Local cache for instant UI; server history reloads after health check for truth

- Practical Pipeline Scripts (optional)
	- Fetch transcripts, clean/store in Mongo, generate graph, backfill embeddings, demo queries

—

## 🧩 Architecture (high level)

- FastAPI backend (templated UI served by Jinja2)
	- `webapp/app.py`: API routes, template rendering, graph endpoints, query streaming
	- `webapp/session_manager.py`: MongoDB-backed session storage
	- `webapp/templates/`: `index.html` (UI), `graph_visualization.html` (D3 graph fragment)
	- `webapp/static/`: `app.js` (front-end logic), `styles.css`

- Front-end
	- Tailwind (via CDN), Marked.js for Markdown
	- D3.js (global CDN + runtime fallback inside the graph template)
	- Inline graph message is inserted after each assistant response; bottom panel disabled by default

- Data & Graph
	- Transcripts in `transcripts/` (JSON)
	- Graph creation and loading via scripts (`scripts/04_*.py`, `scripts/05_*.py`, `scripts/05_graph_loader.py`)

—

## 📁 Repository Structure

```text
trackSabha/
├── Dockerfile
├── LICENSE
├── main.py                      # (Optional runner/entry; see webapp/app.py for the server)
├── pyproject.toml               # Project dependencies / configuration
├── README.md
├── scripts/                     # Data ingestion, cleaning, graph generation/load
│   ├── 01_fetch_yt_transcript.py
│   ├── 02_transcript_to_mongo_uploder.py
│   ├── 03_transcript_clean_from_mongo.py
│   ├── 04_knowledge_graph_generator.py
│   ├── 05_backfill_embeddings.py
│   ├── 05_graph_loader.py
│   └── 06_demo_query.py
├── transcripts/                 # Example transcripts (JSON)
│   └── ...
└── webapp/                      # Web application (FastAPI)
		├── __init__.py
		├── app.py                   # FastAPI app, routes, graph visualization endpoint
		├── prompt.md                # System prompt / instructions
		├── session_manager.py       # MongoDB session storage
		├── static/
		│   ├── app.js              # Front-end chat + graph logic
		│   └── styles.css
		└── templates/
				├── graph_visualization.html  # D3 force layout (embeddable fragment)
				└── index.html                # Main UI
```

—

## 🔌 API Surface (selected)

- `GET /health` — Basic health probe
- `POST /query/stream` — Server-Sent Events (SSE) stream of assistant response
- `GET /session/{session_id}/messages` — Session history
- `GET /session/{session_id}/graph/visualize` — Returns an embeddable HTML fragment with the graph
	- Returns helpful fragments with appropriate status codes (e.g., 404/503) when no graph/initialization issues

—

## 🛠️ Getting Started

### Prerequisites

- Python 3.11+
- A running MongoDB (Atlas or local)
- Windows PowerShell or any shell (examples use PowerShell)

Environment variables (example):

```
# .env (example – adjust to your setup)
MONGODB_URI=mongodb+srv://<user>:<pass>@<cluster>/<db>?retryWrites=true&w=majority
MONGODB_DB=tracksabha
```

> Tip: You can pass these via your shell env or an `--env-file` when using Docker.

### Install and Run (uv)

From the project root:

```powershell
# Install dependencies
uv sync

# Run the app (from the webapp folder)
cd webapp
uv run .\app.py
```

Open http://localhost:8000 and start chatting.

### Run with Docker

```powershell
# Build the image
docker build -t tracksabha .

# Run the container (port-forward and envs)
docker run -p 8000:8000 --env-file .env tracksabha
```

—

## 🗺️ Usage Notes

- Chat interface
	- Type a question and press Enter.
	- After each assistant response, the knowledge graph renders inline as a card.
	- Use the “Hide/Show” toggle on the graph card to minimize it.

- Follow-up suggestions
	- Click any suggestion to send it immediately (safe data attributes, no inline JS issues).

- Session persistence
	- Sessions persist across reloads via MongoDB; localStorage provides instant cache on load.
	- “Clear Chat” wipes messages but preserves your session ID.

- Graph behavior
	- If the graph is empty or the session has no graph yet, the server returns a helpful HTML fragment.
	- D3 is loaded globally; the graph fragment includes a fallback loader to prevent race conditions.

—

## 🔄 Data Pipeline (scripts/)

The `scripts/` directory contains small, focused utilities for data ingestion and graph building:

- `01_fetch_yt_transcript.py` — Fetch YouTube transcripts into `transcripts/`
- `02_transcript_to_mongo_uploder.py` — Load transcripts into MongoDB
- `03_transcript_clean_from_mongo.py` — Clean/normalize stored transcripts
- `04_knowledge_graph_generator.py` — Generate triples/entities/relations from text
- `05_backfill_embeddings.py` — Populate embeddings for vector queries
- `05_graph_loader.py` — Load graph structures into the DB/graph store
- `06_demo_query.py` — Example queries against the pipeline outputs

Run them in order as needed, adjusting environment variables for your setup.

—

## 🧪 Development Notes

- Server uses absolute paths for templates/static/prompt to avoid CWD issues.
- Graph endpoint returns embeddable HTML fragments with proper HTTP status codes and failure messages.
- Client logs graph fetch status and size in the console for quick diagnosis.
- Follow-up suggestions use `data-suggestion` to avoid inline JS escaping bugs.
- Rendered assistant HTML is persisted in message metadata for pixel-perfect rehydration on reload.

—

## 🩺 Troubleshooting

- “Graph doesn’t show”
	- Check the browser console for fetch status and HTML length logs.
	- You may need to ask a question first so the session has context to build a graph.

- “Unexpected UI after changes”
	- Clear `__pycache__` if Python bytecode is stale.
	- Hard-refresh the browser to clear old static assets.

- MongoDB connectivity
	- Verify `MONGODB_URI` and network access to your cluster/instance.

—

## 📄 License

This project is licensed under the terms of the [LICENSE](LICENSE) file.

—

Built for transparency, accountability, and empowerment.