# RAG System with MCP and Ollama

A production-ready Retrieval-Augmented Generation (RAG) system that combines LangChain agents, local LLM (Ollama), and vector storage (ChromaDB) to answer questions from documents.

## Architecture

The system implements a ReAct agent pattern with the following components:

- **User Interface**: CLI-based interaction (main.py)
- **Agent Layer**: ReAct agent with Ollama LLM (rag_agent.py)
- **Tools Layer**: MCP tools for file operations and search (tools.py)
- **Storage Layer**: ChromaDB for vector storage and SentenceTransformers for embeddings (embeddings.py)

## Features

- **ReAct Agent Pattern**: Autonomous reasoning and action execution
- **Local LLM**: Privacy-focused with Ollama (no external API calls)
- **Vector Search**: Semantic search using ChromaDB and sentence embeddings
- **Structured Logging**: Separate logs (stderr) from user output (stdout)
- **Automatic Chunking**: Smart text splitting with word boundary preservation
- **Persistent Storage**: Vector database persists between runs

## Prerequisites

### 1. Install Ollama

```bash
# Download from https://ollama.ai
# Then pull the Mistral model
ollama pull mistral
```

### 2. Python Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Installation

```bash
git clone <repository-url>
cd rag-mcp-local
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python main.py
```

The system will:
1. Prompt for your question
2. Read and process the document
3. Return a contextual answer

### Advanced Usage

```bash
# Run with logs visible
python main.py

# Run with logs saved to file only
python main.py 2>nul  # Windows
python main.py 2>/dev/null  # Linux/Mac

# Save logs to custom file
python main.py 2>debug.log
```

## Project Structure

```
rag-mcp-local/
├── data/
│   └── document.txt          # Sample document (AI overview)
├── db/                       # ChromaDB storage (auto-created)
├── logs/                     # Application logs (auto-created)
│   └── app.log
├── embeddings.py             # Sentence transformer wrapper
├── tools.py                  # MCP tool implementations
├── rag_agent.py              # ReAct agent configuration
├── main.py                   # CLI entry point
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

## Configuration

### Modify LLM Model

Edit `rag_agent.py`:
```python
llm = Ollama(model="mistral", temperature=0)  # Change model here
```

### Adjust Chunk Size

Edit `tools.py`:
```python
max_chars = 500  # Modify chunk size
```

### Change Embedding Model

Edit `embeddings.py`:
```python
MODEL_NAME = "all-MiniLM-L6-v2"  # Change to any sentence-transformers model
```

## How It Works

### 1. Query Processing
User submits a question through the CLI interface.

### 2. Agent Reasoning (ReAct Loop)
The agent follows this pattern:
```
Thought → Action → Observation → Thought → ... → Final Answer
```

### 3. Tool Execution
- **read_file**: Loads document from disk
- **chunk_and_store**: Splits text into 500-char chunks, generates embeddings, stores in ChromaDB
- **search_document**: Performs semantic search, returns top 3 relevant chunks

### 4. Answer Generation
Ollama synthesizes the final answer based on retrieved context.

## Logging

Logs are written to:
- **stderr**: Real-time console output
- **logs/app.log**: Persistent file storage

Log format:
```
[INFO] TOOL CALLED: read_file with path=data/document.txt
[INFO] File read successfully, length=1234 characters
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|----------|
| langchain | Latest | Agent framework |
| langchain-community | Latest | Ollama integration |
| chromadb | Latest | Vector database |
| sentence-transformers | Latest | Text embeddings |
| ollama | Latest | Local LLM |

## Troubleshooting

### Ollama Connection Error
```bash
# Ensure Ollama is running
ollama serve
```

### Model Not Found
```bash
# Pull the required model
ollama pull mistral
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## Performance

- **First run**: ~30s (model download)
- **Subsequent runs**: ~5-10s per query
- **Memory usage**: ~2GB (model + embeddings)
- **Storage**: ~100MB (ChromaDB + models)

## Limitations

- English text only (model limitation)
- Max document size: ~10MB (memory constraint)
- Single document processing per run
- No concurrent query support

## Future Enhancements

- [ ] Multi-document support
- [ ] REST API interface
- [ ] Streaming responses
- [ ] Custom embedding models
- [ ] Query history and caching
- [ ] Docker containerization

## License

MIT License

## Contributing

Contributions welcome! Please follow:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Acknowledgments

- LangChain for the agent framework
- Ollama for local LLM inference
- ChromaDB for vector storage
- Sentence Transformers for embeddings
