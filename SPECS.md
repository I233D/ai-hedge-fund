# AI Hedge Fund - Application Specifications

## 1. Overview

**Name:** AI Hedge Fund
**Version:** 1.0.0
**Purpose:** An AI-powered hedge fund system that uses multiple LLM-based agents to analyze stocks and generate trading decisions. Educational and research use only — does not execute real trades.

---

## 2. System Requirements

### 2.1 Runtime

| Requirement | Specification |
|---|---|
| Python | >= 3.11 |
| Node.js | >= 18 (frontend) |
| OS | Linux, macOS, Windows |
| Docker | Optional (for containerized deployment) |

### 2.2 API Keys (at least one LLM provider required)

| Provider | Environment Variable |
|---|---|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Groq | `GROQ_API_KEY` |
| DeepSeek | `DEEPSEEK_API_KEY` |
| Google Gemini | `GOOGLE_API_KEY` |
| xAI (Grok) | `XAI_API_KEY` |
| GigaChat | `GIGACHAT_API_KEY` |
| Azure OpenAI | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` |
| Financial Datasets | `FINANCIAL_DATASETS_API_KEY` |

Local LLM support available via **Ollama** (no API key needed).

---

## 3. Architecture

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Interfaces                       │
│   ┌──────────────┐    ┌──────────────────────────────┐  │
│   │   CLI (src/)  │    │  Web App (app/)              │  │
│   │   main.py     │    │  FastAPI + React/TypeScript  │  │
│   └──────┬───────┘    └──────────────┬───────────────┘  │
│          │                           │                   │
│   ┌──────▼───────────────────────────▼───────────────┐  │
│   │           LangGraph Agent Workflow                │  │
│   │  ┌─────────────────────────────────────────────┐ │  │
│   │  │         17 Analyst Agents (parallel)         │ │  │
│   │  └─────────────────┬───────────────────────────┘ │  │
│   │                    ▼                              │  │
│   │  ┌──────────────────────────────────────┐        │  │
│   │  │  Risk Manager → Portfolio Manager    │        │  │
│   │  └──────────────────────────────────────┘        │  │
│   └──────────────────────┬───────────────────────────┘  │
│                          ▼                               │
│   ┌──────────────────────────────────────────────────┐  │
│   │              Data Layer                           │  │
│   │  Financial Datasets API  |  In-Memory Cache       │  │
│   └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Component Breakdown

| Component | Location | Purpose |
|---|---|---|
| CLI | `src/main.py` | Command-line interface for running analysis |
| Backtester | `src/backtester.py` | Historical strategy simulation |
| Agents | `src/agents/` | 17 analyst + 2 infrastructure agents |
| Data Models | `src/data/models.py` | Pydantic schemas for financial data |
| Data Cache | `src/data/cache.py` | In-memory caching for API responses |
| API Tools | `src/tools/api.py` | Financial Datasets API integration |
| Graph State | `src/graph/state.py` | LangGraph workflow state management |
| LLM Config | `src/llm/models.py` | Multi-provider LLM configuration |
| Utilities | `src/utils/` | Analyst registry, display, progress |
| Backend API | `app/backend/` | FastAPI REST API |
| Frontend | `app/frontend/` | React/TypeScript SPA |

---

## 4. Functional Specifications

### 4.1 Agent System

The system employs **17 analyst agents**, each modeled after a famous investor's philosophy:

| Agent | Style | Key Focus |
|---|---|---|
| Warren Buffett | Value investing | Competitive moats, intrinsic value |
| Charlie Munger | Quality business | Mental models, business quality |
| Ben Graham | Margin of safety | Net-net, deep value |
| Peter Lynch | GARP | Growth at reasonable price |
| Phil Fisher | Scuttlebutt | Qualitative research |
| Cathie Wood | Disruptive growth | Innovation, exponential growth |
| Michael Burry | Contrarian | Short selling, market inefficiencies |
| Bill Ackman | Activist | Catalysts, corporate governance |
| Stanley Druckenmiller | Macro | Macro trends, asymmetric risk/reward |
| Aswath Damodaran | Valuation | DCF, cost of capital |
| Rakesh Jhunjhunwala | Emerging markets | High-growth opportunities |
| Mohnish Pabrai | Dhandho | Low-risk, high-return bets |
| Fundamentals Agent | Quantitative | Financial statements analysis |
| Technicals Agent | Technical analysis | Price patterns, indicators |
| Sentiment Agent | Market sentiment | Market mood, positioning |
| News Sentiment Agent | NLP | News analysis, event-driven |
| Growth Agent | Growth metrics | Revenue/earnings growth trends |

**Infrastructure Agents:**

| Agent | Role |
|---|---|
| Risk Manager | Position sizing, exposure limits, portfolio risk |
| Portfolio Manager | Final trade decisions, signal aggregation |

### 4.2 Signal Pipeline

```
1. User Input (tickers, date range, analyst selection)
       │
2. Analyst Agents run in parallel
       │  Each produces: signal (bullish/bearish/neutral)
       │                 confidence (0-100%)
       │                 reasoning (text)
       │
3. Risk Manager evaluates aggregate risk
       │
4. Portfolio Manager generates final decisions
       │  Output: BUY / SELL / SHORT / COVER / HOLD
       │          quantity, reasoning
       │
5. Results displayed (CLI table or web UI)
```

### 4.3 Backtesting Engine

| Feature | Specification |
|---|---|
| Simulation | Day-by-day over configurable date ranges |
| Positions | Long and short support |
| Margin | Configurable margin requirements |
| Slippage | Spread/slippage modeling |
| Metrics | Sharpe ratio, Sortino ratio, max drawdown, total return |
| Benchmark | SPY comparison |
| Output | Tabular results with per-trade breakdown |

### 4.4 Data Integration

| Data Type | Source | Endpoint |
|---|---|---|
| Stock Prices | Financial Datasets API | Historical OHLCV |
| Financial Metrics | Financial Datasets API | Income, balance sheet, cash flow |
| Insider Trades | Financial Datasets API | SEC Form 4 filings |
| News | Financial Datasets API | Company news articles |
| Market Data | Financial Datasets API | Index and benchmark data |

**Free tier** supports: AAPL, GOOGL, MSFT, NVDA, TSLA. Other tickers require a paid API key.

### 4.5 Web Application

**Backend (FastAPI):**

| Feature | Details |
|---|---|
| Framework | FastAPI with Uvicorn |
| Database | SQLAlchemy ORM + Alembic migrations |
| API Docs | Auto-generated at `/docs` (Swagger) |
| Routes | `/api/` prefixed REST endpoints |
| Services | Business logic layer for agent orchestration |

**Frontend (React):**

| Feature | Details |
|---|---|
| Framework | React 18 + TypeScript |
| Build Tool | Vite |
| UI Library | Shadcn UI + Radix UI primitives |
| Styling | Tailwind CSS |
| Visualization | @xyflow/react (flow-based agent graph) |
| Theming | Dark/light mode via next-themes |
| Icons | Lucide React |

---

## 5. Tech Stack Summary

### 5.1 Core Dependencies

| Category | Package | Version |
|---|---|---|
| Agent Framework | LangChain | ^0.3.7 |
| Agent Orchestration | LangGraph | 0.2.56 |
| Data Processing | pandas | ^2.1.0 |
| Data Processing | numpy | ^1.24.0 |
| Data Validation | pydantic | ^2.4.2 |
| Web Framework | FastAPI | ^0.104.0 |
| ORM | SQLAlchemy | ^2.0.22 |
| Migrations | Alembic | ^1.12.0 |
| HTTP Client | httpx | ^0.27.0 |
| CLI | questionary, rich, colorama | various |
| Environment | python-dotenv | 1.0.0 |

### 5.2 LLM Provider SDKs

| Provider | Package | Version |
|---|---|---|
| OpenAI | langchain-openai | ^0.3.5 |
| Anthropic | langchain-anthropic | 0.3.5 |
| Groq | langchain-groq | 0.2.3 |
| DeepSeek | langchain-deepseek | ^0.1.2 |
| Google Gemini | langchain-google-genai | ^2.0.11 |
| xAI | langchain-xai | ^0.2.5 |
| GigaChat | langchain-gigachat | ^0.3.12 |
| Ollama (local) | langchain-ollama | 0.3.6 |

### 5.3 Dev Dependencies

| Tool | Purpose | Version |
|---|---|---|
| pytest | Testing | ^7.4.0 |
| black | Code formatting | ^23.7.0 |
| isort | Import sorting | ^5.12.0 |
| flake8 | Linting | ^6.1.0 |

---

## 6. Non-Functional Requirements

### 6.1 Performance

- Agent execution is parallelized via LangGraph for concurrent analysis
- In-memory caching reduces redundant API calls within a session
- Rate limiting with exponential backoff for external API calls

### 6.2 Scalability

- Modular agent design — new analyst agents can be added by creating a new file in `src/agents/` and registering in `src/utils/analysts.py`
- Multi-provider LLM support allows switching models without code changes
- Docker Compose supports scaling individual services

### 6.3 Security

- API keys stored in `.env` file (not committed to VCS)
- `.env.example` provided as a template
- No real trading execution — simulation only
- No user authentication in current web app (local use)

### 6.4 Extensibility

- **Add new agent:** Create `src/agents/<name>.py`, register in `src/utils/analysts.py`
- **Add new LLM provider:** Add to `src/llm/models.py` and corresponding `*_models.json`
- **Add new data source:** Extend `src/tools/api.py`
- **Add new backtest strategy:** Extend `src/backtesting/engine.py`

---

## 7. Testing

| Test Category | Location | Coverage |
|---|---|---|
| Backtest Engine | `tests/backtesting/test_engine.py` | Engine orchestration |
| Portfolio Logic | `tests/backtesting/test_portfolio.py` | Position tracking, P&L |
| Metrics | `tests/backtesting/test_metrics.py` | Sharpe, Sortino, drawdown |
| Integration | `tests/backtesting/integration/` | End-to-end strategy tests |
| Fixtures | `tests/fixtures/` | Sample financial data |

Run tests with:
```bash
poetry run pytest
```

---

## 8. Deployment

### 8.1 Local Development

```bash
# Install dependencies
poetry install

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run CLI
poetry run python src/main.py --ticker AAPL,MSFT,NVDA

# Run web app
cd app && ./run.sh
```

### 8.2 Docker

```bash
cd docker
docker-compose up hedge-fund          # CLI mode
docker-compose up backtester          # Backtest mode
docker-compose up hedge-fund-ollama   # With local LLMs
```

---

## 9. Disclaimer

This project is for **educational and research purposes only**. It is not intended as financial advice or a recommendation to trade. No real money should be risked based on this system's outputs. The creators assume no liability for financial losses.
