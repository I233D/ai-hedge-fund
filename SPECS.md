# AI Hedge Fund - Application Specifications

## 1. Overview

**Name:** AI Hedge Fund
**Version:** 1.0.0
**Purpose:** An AI-powered hedge fund system that uses multiple LLM-based agents and quantitative strategies to analyze stocks and futures, generate trading decisions, and backtest performance. Educational and research use only — does not execute real trades.

---

## 2. System Requirements

### 2.1 Runtime

| Requirement | Specification |
|---|---|
| Python | >= 3.11 |
| Node.js | >= 18 (frontend) |
| OS | Linux, macOS, Windows |
| Docker | Optional (for containerized deployment) |
| IBKR Client Portal | Optional (for 24h ES/NQ/SPX futures feed) |

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

### 3.1 Full System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACES                                     │
│                                                                                  │
│  ┌────────────────────┐          ┌──────────────────────────────────────────┐    │
│  │  CLI (src/main.py) │          │  Web App (app/)                          │    │
│  │                    │          │                                          │    │
│  │  --ticker AAPL,ES  │          │  ┌──────────────┐  ┌─────────────────┐  │    │
│  │  --start-date ...  │          │  │ React/TS SPA │  │ FastAPI Backend │  │    │
│  │  --analysts ...    │          │  │ Shadcn UI    │◄─┤ /api/...        │  │    │
│  │  --show-reasoning  │          │  │ @xyflow      │  │ SQLAlchemy      │  │    │
│  │  --ollama          │          │  │ Tailwind CSS │  │ Alembic         │  │    │
│  └────────┬───────────┘          │  └──────────────┘  └────────┬────────┘  │    │
│           │                      └──────────────────────────────┼──────────┘    │
└───────────┼─────────────────────────────────────────────────────┼────────────────┘
            │                                                     │
            ▼                                                     ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                      DUAL AGENT SYSTEM                                           │
│                                                                                  │
│  ┌─────────────────────────────┐    ┌──────────────────────────────────────┐    │
│  │  LLM-Based Agents           │    │  Quantitative Strategies             │    │
│  │  (src/agents/)              │    │  (src/strategies/)                   │    │
│  │                             │    │                                      │    │
│  │  17 analyst agents using    │    │  17 investor strategies using        │    │
│  │  LangGraph + LLM reasoning  │    │  pure math, zero LLM calls          │    │
│  │  for detailed narratives    │    │  < 1ms total sweep                   │    │
│  │                             │    │                                      │    │
│  │  + Risk Manager             │    │  + RiskManagerStrategy (vol guard)   │    │
│  │  + Portfolio Manager        │    │  + PortfolioManagerStrategy (gate)   │    │
│  └──────────────┬──────────────┘    └──────────────┬───────────────────────┘    │
│                 │                                   │                            │
│                 ▼                                   ▼                            │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │  SignalAggregator (confidence-weighted parallel reduce)                   │   │
│  │  BUY / SELL / SHORT / COVER / HOLD + confidence 0.0–1.0                  │   │
│  └──────────────────────────────┬───────────────────────────────────────────┘   │
│                                 │                                               │
└─────────────────────────────────┼───────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                       DATA FEED LAYER                                            │
│                                                                                  │
│  ┌──────────────────────┐  ┌──────────────────┐  ┌────────────────────────┐    │
│  │ Financial Datasets   │  │ IBKR Client      │  │ MacroDataProvider      │    │
│  │ API (src/tools/)     │  │ Portal API       │  │ (src/strategies/)      │    │
│  │                      │  │ (src/feeds/)     │  │                        │    │
│  │ Prices, Metrics,     │  │                  │  │ VIX, 10Y, Oil,        │    │
│  │ Insider Trades,      │  │ ES/NQ/SPX 24h   │  │ DXY, Gold, CPI,       │    │
│  │ News, Line Items     │  │ Adaptive fresh   │  │ GDP, Credit Spread    │    │
│  │                      │  │ CircuitBreaker   │  │                        │    │
│  │ + In-Memory Cache    │  │ + Quote Cache    │  │ + Domain Multipliers   │    │
│  └──────────────────────┘  └──────────────────┘  └────────────────────────┘    │
│                                                                                  │
│  DataFeedOrchestrator (parallel multi-feed → FeedBundle per ticker)              │
│  TickerWeights (ES→macro agents, NQ→growth agents, equity→standard)             │
│                                                                                  │
└──────────────────────────────┬───────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                       BACKTESTING & OUTPUT                                        │
│                                                                                  │
│  Backtester (src/backtesting/) + SPXOptionsTrainer (src/strategies/trainer.py)   │
│  Benchmarks: SPY, QQQ, ES, NQ  |  Metrics: Sharpe, Sortino, MaxDD              │
│                                                                                  │
│  ┌───────────────┐  ┌──────────────────┐  ┌──────────────────────────────┐      │
│  │ CLI Table      │  │ React Dashboard   │  │ TradingAlert + POST /ingest │      │
│  │ (rich/tabulate)│  │ (SSE streaming)   │  │ (webhook / notification)    │      │
│  └───────────────┘  └──────────────────┘  └──────────────────────────────┘      │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Component Breakdown

| Component | Location | Purpose |
|---|---|---|
| **CLI** | `src/main.py` | Command-line interface for running analysis |
| **Backtester** | `src/backtester.py` | Historical strategy simulation |
| **LLM Agents** | `src/agents/` | 17 analyst + 2 infrastructure agents (LLM-based) |
| **Quant Strategies** | `src/strategies/` | 17 investor + 2 infra strategies (pure math) |
| **Data Feeds** | `src/feeds/` | IBKR 24h feed, orchestrator, ticker weights |
| **Data Models** | `src/data/models.py` | Pydantic schemas for financial data |
| **Data Cache** | `src/data/cache.py` | In-memory caching for API responses |
| **API Tools** | `src/tools/api.py` | Financial Datasets API integration |
| **Graph State** | `src/graph/state.py` | LangGraph workflow state management |
| **LLM Config** | `src/llm/models.py` | Multi-provider LLM configuration |
| **Utilities** | `src/utils/` | Analyst registry, display, progress |
| **Backend API** | `app/backend/` | FastAPI REST API with SSE streaming |
| **Frontend** | `app/frontend/` | React/TypeScript SPA (VS Code-style IDE) |

---

## 4. Functional Specifications

### 4.1 LLM-Based Agent System (Original)

The system employs **17 analyst agents**, each modeled after a famous investor's philosophy, powered by LLM reasoning:

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

### 4.2 Quantitative Strategy System (Production)

Pure-math, zero-LLM strategies running in parallel via `ThreadPoolExecutor`. Total 19-agent sweep completes in < 1ms.

| # | Strategy | Style | Primary Signals |
|---|---|---|---|
| 1 | BuffettStrategy | Value | ROE + moat + intrinsic gap + low debt |
| 2 | MungerStrategy | Inversion | Red flag detection + ROIC + quality |
| 3 | BurryStrategy | Contrarian short | Slope + vol + overvaluation / deep value buy |
| 4 | CathieWoodStrategy | Growth | Revenue/earnings growth + regime |
| 5 | DalioStrategy | Risk parity | Low vol preference + balanced regime |
| 6 | SorosStrategy | Reflexivity | Slope + volume breakout |
| 7 | AckmanStrategy | Activist | Gross margin + intrinsic gap + volume |
| 8 | DruckenmillerStrategy | Macro timing | Regime + momentum |
| 9 | PeterLynchStrategy | GARP | PEG ratio + earnings growth |
| 10 | JimSimonsStrategy | Quant pattern | RSI + vol + SMA cross + volume |
| 11 | CarlIcahnStrategy | Activist value | Intrinsic gap + P/B + insider buys |
| 12 | DavidTepperStrategy | Distressed | Bearish regime + sound fundamentals |
| 13 | BillMillerStrategy | Contrarian value | Negative sentiment + beaten down |
| 14 | JohnPaulsonStrategy | Macro event | Extreme vol + directional bet |
| 15 | PaulTudorJonesStrategy | Trend following | SMA cross + slope |
| 16 | KenGriffinStrategy | Vol arb | Vol mean reversion + volume |
| 17 | SteveCohenStrategy | Event driven | Insider + volume + news catalyst |

**Infrastructure Strategies:**

| Strategy | Role |
|---|---|
| RiskManagerStrategy | Volatility guard — vol > 0.70 → COVER (0.95 confidence veto) |
| PortfolioManagerStrategy | Quality gate — validates fundamentals before position sizing |

**Signal Aggregation:**
- Confidence-weighted parallel reduce across 17 investor agents
- Risk Manager veto: COVER + confidence >= 0.90 → force override
- Infrastructure agents excluded from directional voting
- Domain multipliers: per-agent data weighting (Burry.slope × 2.0, Druckenmiller.regime × 1.8)
- Macro adjustments: VIX/rates/oil/DXY/gold inject ±0.10 confidence shift

### 4.3 Signal Pipeline

```
1. User Input (tickers, date range, analyst selection)
       │
       ├──────────────────────────────────────────────────┐
       │                                                  │
2a. LLM Agents run in parallel              2b. Quant Strategies run in parallel
       │  signal + confidence + reasoning          │  signal + confidence (pure math)
       │                                           │
3. Risk Manager evaluates aggregate risk     3. RiskManagerStrategy (vol guard)
       │                                           │
4. Portfolio Manager final decisions         4. SignalAggregator (weighted vote)
       │                                           │
       └──────────────────┬───────────────────────┘
                          │
5. Results: BUY / SELL / SHORT / COVER / HOLD
   + quantity, confidence, reasoning
       │
6. CLI table or React dashboard (SSE streaming)
```

### 4.4 Backtesting Engine

| Feature | Specification |
|---|---|
| Simulation | Day-by-day over configurable date ranges |
| Positions | Long and short support |
| Margin | Configurable margin requirements |
| Slippage | Spread/slippage modeling |
| Metrics | Sharpe ratio, Sortino ratio, max drawdown, total return |
| Benchmarks | SPY, QQQ, ES, NQ (ticker-aware) |
| Output | Tabular results with per-trade breakdown |
| Training | SPXOptionsTrainer for per-agent Sharpe evaluation |

### 4.5 Data Integration

| Data Type | Source | Details |
|---|---|---|
| Stock Prices | Financial Datasets API | Historical OHLCV (cached) |
| Financial Metrics | Financial Datasets API | P/E, ROE, margins, growth |
| Insider Trades | Financial Datasets API | SEC Form 4 filings |
| News | Financial Datasets API | Company news articles |
| **Futures Quotes** | **IBKR Client Portal API** | **ES/NQ/SPX 24h real-time** |
| **Macro Data** | **MacroDataProvider** | **VIX, 10Y, oil, DXY, gold, CPI, GDP** |

**Free tier** supports: AAPL, GOOGL, MSFT, NVDA, TSLA. Other tickers require a paid API key.

### 4.6 IBKR ES/NQ/SPX 24h Pre-Market Feed

| Component | File | Purpose |
|---|---|---|
| IBKRClient | `src/feeds/ibkr_client.py` | Client Portal API wrapper, session mgmt, retry, conid mapping |
| IBKRRealTimeFeed | `src/feeds/ibkr_feed.py` | Cached feed + adaptive freshness + circuit breaker + markers |
| CircuitBreaker | `src/feeds/circuit_breaker.py` | CLOSED/OPEN/HALF_OPEN protection against stale feeds |
| AdaptiveFreshness | `src/feeds/freshness.py` | VIX-scaled, session-aware freshness bounds |
| DataFeedOrchestrator | `src/feeds/orchestrator.py` | Parallel multi-feed → FeedBundle aggregation |
| TickerWeights | `src/feeds/ticker_weights.py` | Per-instrument agent confidence multipliers |
| Benchmarks | `src/feeds/benchmarks.py` | ES→SPY, NQ→QQQ comparison with Sharpe/Sortino/MaxDD |

**Adaptive Freshness Bounds:**

```
Symbol   Session       VIX    Bound
──────   ──────────   ────   ─────
ES       RTH          20     17s
ES       pre-market   20     34s
ES       overnight    20     51s
NQ       pre-market   35     ~59s
SPX      RTH          20     60s
VIX scaling: ×(VIX/20), clamped [0.5, 2.0]
Min: 15s, Max: 900s
```

**Ticker Weights (agent confidence multipliers):**

| Ticker | Boosted Agents | Reduced Agents |
|---|---|---|
| ES | Druckenmiller ×1.5, PaulTudorJones ×1.5, Dalio ×1.4, KenGriffin ×1.4 | PeterLynch ×0.6, CarlIcahn ×0.5 |
| NQ | CathieWood ×1.5, JimSimons ×1.3, Druckenmiller ×1.3 | CarlIcahn ×0.5, Buffett ×0.7 |
| SPX | Druckenmiller ×1.3, PaulTudorJones ×1.3, Dalio ×1.2 | CarlIcahn ×0.7, CathieWood ×0.9 |
| Equity | All agents ×1.0 (standard weighting) | — |

---

## 5. Web Application (UI)

### 5.1 Frontend Architecture (VS Code-Style IDE)

```
┌──────────────────────────────────────────────────────────────┐
│  Top Bar (title, sidebar toggles, settings)                  │
├──────────┬───────────────────────────────────┬───────────────┤
│          │                                   │               │
│  Left    │    Main Content Area              │    Right      │
│  Sidebar │                                   │    Sidebar    │
│  (280px) │    Tab Bar                        │    (280px)    │
│          │    ┌────────────────────────┐     │               │
│  Flow    │    │  ReactFlow Canvas      │     │  Component    │
│  List    │    │  (drag-drop agents)    │     │  Palette      │
│  ─────── │    │                        │     │  ───────────  │
│  Search  │    │  Nodes: analyst agents │     │  Draggable    │
│  Create  │    │  Edges: data flow      │     │  agent cards  │
│  Edit    │    │  Auto-save + undo/redo │     │  grouped by   │
│  Delete  │    │                        │     │  category     │
│  Dupe    │    └────────────────────────┘     │               │
│          │                                   │               │
├──────────┴───────────────────────────────────┴───────────────┤
│  Bottom Panel (300px, collapsible)                           │
│  ┌─────────┬──────────┬─────────┬──────────┬───────────┐    │
│  │ Output  │ Backtest │ Debug   │ Terminal │ Problems  │    │
│  │         │ Results  │ Console │          │           │    │
│  └─────────┴──────────┴─────────┴──────────┴───────────┘    │
│  Agent reasoning | Trade decisions | SSE stream output       │
└──────────────────────────────────────────────────────────────┘

Keyboard Shortcuts:
  Cmd+B  Toggle left sidebar
  Cmd+I  Toggle right sidebar
  Cmd+J  Toggle bottom panel
  Cmd+O  Open flow
```

### 5.2 Frontend Tech Stack

| Feature | Technology |
|---|---|
| Framework | React 18.2.0 + TypeScript 5.3.3 |
| Build Tool | Vite 5.0.12 |
| Flow Editor | @xyflow/react 12.5.1 (node-based visual editor) |
| UI Library | shadcn-ui 0.9.5 + Radix UI (accordion, checkbox, dialog, tabs, tooltip, popover) |
| Styling | Tailwind CSS 3.4.1 + tailwindcss-animate |
| Theming | next-themes (dark/light mode) |
| Panels | react-resizable-panels |
| Code Display | react-syntax-highlighter |
| Notifications | sonner (toast) |
| Icons | Lucide React |

### 5.3 Frontend Component Hierarchy

```
App.tsx
└── Layout.tsx (ThemeProvider, NodeProvider, FlowProvider, TabsProvider, LayoutProvider)
    ├── top-bar.tsx (VS Code-style toolbar)
    ├── left-sidebar.tsx (collapsible, 280px)
    │   ├── flow-list.tsx (grid of saved flows)
    │   ├── flow-item.tsx / flow-item-group.tsx
    │   ├── flow-create-dialog.tsx / flow-edit-dialog.tsx
    │   ├── flow-context-menu.tsx (right-click)
    │   └── search-box.tsx
    ├── tab-bar.tsx + tab-content.tsx
    │   └── Flow.tsx (ReactFlow canvas with auto-save + history)
    ├── right-sidebar.tsx (collapsible, 280px)
    │   ├── component-list.tsx (available agents)
    │   ├── component-item.tsx (draggable)
    │   └── component-item-group.tsx
    ├── bottom-panel.tsx (collapsible, 300px)
    │   ├── output-tab.tsx / regular-output.tsx
    │   ├── backtest-output.tsx
    │   ├── debug-console-tab.tsx
    │   ├── terminal-tab.tsx
    │   ├── problems-tab.tsx
    │   └── reasoning-content.tsx
    └── settings/ (dialog)
        ├── api-keys.tsx
        ├── models.tsx (cloud.tsx + ollama.tsx)
        └── appearance.tsx
```

### 5.4 Frontend State Management

| Context | File | State |
|---|---|---|
| NodeProvider | `contexts/node-context.tsx` | Node types, available agents |
| FlowProvider | `contexts/flow-context.tsx` | Current flow, CRUD operations |
| LayoutProvider | `contexts/layout-context.tsx` | Sidebar/panel visibility |
| TabsProvider | `contexts/tabs-context.tsx` | Tab management |

### 5.5 Frontend Services (API Layer)

| Service | File | Endpoints |
|---|---|---|
| Flow API | `services/flow-api.ts` | CRUD for flows |
| Hedge Fund API | `services/hedge-fund-api.ts` | Run analysis, backtest |
| API Keys API | `services/api-keys-api.ts` | Key management |
| Ollama API | `services/ollama-api.ts` | Local model management |
| Sidebar Storage | `services/sidebar-storage.ts` | localStorage persistence |

---

## 6. Backend API

### 6.1 Route Map

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/hedge-fund/run` | Execute analysis (SSE streaming) |
| `POST` | `/hedge-fund/backtest` | Run backtest (SSE streaming) |
| `GET` | `/hedge-fund/agents` | List available analysts |
| `POST` | `/flows/` | Create flow |
| `GET` | `/flows/` | List all flows |
| `GET` | `/flows/{id}` | Get flow with nodes/edges |
| `PUT` | `/flows/{id}` | Update flow |
| `DELETE` | `/flows/{id}` | Delete flow |
| `POST` | `/flows/{id}/duplicate` | Clone flow |
| `GET` | `/flows/search/{name}` | Search flows |
| `POST` | `/flows/{id}/runs` | Create run |
| `GET` | `/flows/{id}/runs` | List runs (paginated) |
| `GET` | `/flows/{id}/runs/active` | Get active run |
| `GET` | `/flows/{id}/runs/latest` | Get latest run |
| `POST` | `/api-keys/` | Create/update key |
| `GET` | `/api-keys/` | List keys (no values) |
| `DELETE` | `/api-keys/{provider}` | Delete key |
| `POST` | `/api-keys/bulk` | Bulk create/update |
| `GET` | `/language-models/` | List all models |
| `GET` | `/language-models/providers` | Group by provider |
| `GET` | `/ollama/status` | Check Ollama status |
| `POST` | `/ollama/start` | Start Ollama server |
| `POST` | `/ollama/models/download` | Download model (SSE) |
| `DELETE` | `/ollama/models/{name}` | Delete model |
| `GET` | `/ping` | Health check (SSE) |

### 6.2 Backend Services

| Service | File | Purpose |
|---|---|---|
| Agent Service | `services/agent_service.py` | Create agent functions from analyst configs |
| Graph Service | `services/graph.py` | Build LangGraph StateGraph from ReactFlow |
| Portfolio Service | `services/portfolio.py` | Portfolio state management |
| Backtest Service | `services/backtest_service.py` | Historical strategy simulation |
| API Key Service | `services/api_key_service.py` | Key CRUD with database |
| Ollama Service | `services/ollama_service.py` | Local LLM lifecycle management |

### 6.3 Database Schema (SQLAlchemy + SQLite)

| Table | Key Fields |
|---|---|
| `HedgeFundFlow` | id, name, description, nodes_json, edges_json, viewport, tags, is_template |
| `HedgeFundFlowRun` | id, flow_id (FK), status, request_data, results, timestamps |
| `HedgeFundFlowRunCycle` | id, run_id (FK), analyst_signals, trades, portfolio_snapshots, metrics |
| `ApiKey` | id, provider, key_value, is_active, last_used |

### 6.4 Streaming (Server-Sent Events)

| Event Type | Payload |
|---|---|
| `StartEvent` | Execution beginning signal |
| `ProgressUpdateEvent` | Agent name, ticker, status, analysis text |
| `ErrorEvent` | Error message |
| `CompleteEvent` | Final decisions and data |

---

## 7. Tech Stack Summary

### 7.1 Core Dependencies

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

### 7.2 LLM Provider SDKs

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

### 7.3 Frontend Dependencies

| Category | Package | Version |
|---|---|---|
| Framework | React | 18.2.0 |
| Build | Vite | 5.0.12 |
| Language | TypeScript | 5.3.3 |
| Flow Editor | @xyflow/react | 12.5.1 |
| UI Components | shadcn-ui | 0.9.5 |
| Styling | Tailwind CSS | 3.4.1 |
| Theming | next-themes | latest |
| Panels | react-resizable-panels | latest |
| Notifications | sonner | latest |

### 7.4 Dev Dependencies

| Tool | Purpose | Version |
|---|---|---|
| pytest | Testing | ^7.4.0 |
| black | Code formatting | ^23.7.0 |
| isort | Import sorting | ^5.12.0 |
| flake8 | Linting | ^6.1.0 |

---

## 8. Non-Functional Requirements

### 8.1 Performance

- LLM agents parallelized via LangGraph for concurrent analysis
- Quantitative strategies complete full 19-agent sweep in < 1ms
- IBKR feed adds < 0.4ms latency (zero-copy cached)
- In-memory caching reduces redundant API calls within a session
- Rate limiting with exponential backoff for external API calls
- Adaptive freshness bounds prevent unnecessary IBKR re-fetches

### 8.2 Scalability

- Modular agent design — new analyst agents can be added by creating a new file in `src/agents/` and registering in `src/utils/analysts.py`
- New quantitative strategies: subclass `StrategyAgent`, add to `ALL_INVESTOR_STRATEGIES`
- Multi-provider LLM support allows switching models without code changes
- Docker Compose supports scaling individual services
- DataFeedOrchestrator supports parallel multi-ticker feed resolution

### 8.3 Reliability

- CircuitBreaker protects against IBKR feed failures (CLOSED/OPEN/HALF_OPEN)
- Zero-fill safety: all 17 quant agents return HOLD on fallback data
- Stale cache fallback: returns last known quote when live feed fails
- Session-aware freshness: pre-market/overnight bounds automatically relaxed

### 8.4 Security

- API keys stored in `.env` file (not committed to VCS) and database
- `.env.example` provided as a template
- IBKR Client Portal uses local HTTPS with self-signed certs
- No real trading execution — simulation only
- No user authentication in current web app (local use)

### 8.5 Extensibility

- **Add new LLM agent:** Create `src/agents/<name>.py`, register in `src/utils/analysts.py`
- **Add new quant strategy:** Subclass `StrategyAgent` in `src/strategies/investors.py`, add to `ALL_INVESTOR_STRATEGIES`
- **Add new LLM provider:** Add to `src/llm/models.py` and corresponding `*_models.json`
- **Add new data source:** Extend `src/tools/api.py` or add new feed in `src/feeds/`
- **Add new futures ticker:** Add conid to `FUTURES_CONIDS`, weights to `ticker_weights.py`
- **Add new backtest strategy:** Extend `src/backtesting/engine.py`
- **Add new macro indicator:** Extend `MacroData` in `src/strategies/macro.py`

---

## 9. Testing

| Test Category | Location | Tests | Coverage |
|---|---|---|---|
| Backtest Engine | `tests/backtesting/test_engine.py` | — | Engine orchestration |
| Portfolio Logic | `tests/backtesting/test_portfolio.py` | — | Position tracking, P&L |
| Metrics | `tests/backtesting/test_metrics.py` | — | Sharpe, Sortino, drawdown |
| Integration | `tests/backtesting/integration/` | — | End-to-end strategy tests |
| **Quant Strategies** | `tests/strategies/test_strategies.py` | **65** | All 17 investors, aggregator, runner, 24 edge cases |
| **Macro + Trainer** | `tests/strategies/test_macro_and_trainer.py` | **16** | Macro adjustments, domain multipliers, SPX trainer |
| **IBKR Feeds** | `tests/feeds/test_feeds.py` | **70** | Client, circuit breaker, freshness, orchestrator, benchmarks, 24 edge cases |
| Fixtures | `tests/fixtures/` | — | Sample financial data |

**Total: 151 tests, all passing in ~1.25s**

Run tests with:
```bash
poetry run pytest                          # all tests
poetry run pytest tests/strategies/ -v     # quant strategies only
poetry run pytest tests/feeds/ -v          # IBKR feeds only
```

---

## 10. Deployment

### 10.1 Local Development

```bash
# Install dependencies
poetry install

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run CLI (equities)
poetry run python src/main.py --ticker AAPL,MSFT,NVDA

# Run CLI (show reasoning)
poetry run python src/main.py --ticker AAPL --show-reasoning

# Run CLI (local Ollama models)
poetry run python src/main.py --ticker AAPL --ollama

# Run backtester
poetry run python src/backtester.py --ticker AAPL,MSFT,NVDA

# Run web app
cd app && ./run.sh
```

### 10.2 Docker

```bash
cd docker
docker-compose up hedge-fund          # CLI mode
docker-compose up backtester          # Backtest mode
docker-compose up hedge-fund-ollama   # With local LLMs
```

### 10.3 IBKR Setup (Optional – for 24h Futures)

```bash
# 1. Start IBKR Client Portal Gateway
cd ~/clientportal.gw && bin/run.sh root/conf.yaml

# 2. Authenticate via browser at https://localhost:5000

# 3. The app auto-connects via IBKRClient when trading ES/NQ/SPX
```

---

## 11. File Map

```
ai-hedge-fund/
├── SPECS.md
├── README.md
├── pyproject.toml
├── .env.example
│
├── src/
│   ├── main.py                          # CLI entry point
│   ├── backtester.py                    # Backtest entry point
│   │
│   ├── agents/                          # LLM-based agents (17 + 2)
│   │   ├── warren_buffett.py            #   Value investing
│   │   ├── charlie_munger.py            #   Quality + inversion
│   │   ├── michael_burry.py             #   Contrarian shorting
│   │   ├── cathie_wood.py               #   Disruptive growth
│   │   ├── ben_graham.py                #   Margin of safety
│   │   ├── bill_ackman.py               #   Activist investing
│   │   ├── peter_lynch.py               #   GARP
│   │   ├── phil_fisher.py               #   Scuttlebutt research
│   │   ├── aswath_damodaran.py          #   DCF valuation
│   │   ├── stanley_druckenmiller.py     #   Macro investing
│   │   ├── rakesh_jhunjhunwala.py       #   Emerging markets
│   │   ├── mohnish_pabrai.py            #   Dhandho investing
│   │   ├── fundamentals.py              #   Financial statements
│   │   ├── technicals.py                #   Technical analysis
│   │   ├── sentiment.py                 #   Market sentiment
│   │   ├── news_sentiment.py            #   News NLP
│   │   ├── growth_agent.py              #   Growth metrics
│   │   ├── risk_manager.py              #   Risk management
│   │   └── portfolio_manager.py         #   Portfolio decisions
│   │
│   ├── strategies/                      # Quantitative strategies (17 + 2)
│   │   ├── __init__.py                  #   Module exports
│   │   ├── base.py                      #   Signal, TradingAction, StrategyAgent
│   │   ├── snapshot.py                  #   FinancialSnapshot + build_snapshot()
│   │   ├── investors.py                 #   17 concrete investor strategies
│   │   ├── aggregator.py               #   Confidence-weighted signal aggregator
│   │   ├── risk_guard.py               #   RiskManager + PortfolioManager strategies
│   │   ├── runner.py                    #   Parallel strategy runner (ThreadPool)
│   │   ├── macro.py                     #   MacroData + domain multipliers
│   │   └── trainer.py                   #   SPXOptionsTrainer (per-agent Sharpe)
│   │
│   ├── feeds/                           # IBKR 24h data feeds
│   │   ├── __init__.py                  #   Module exports
│   │   ├── ibkr_client.py              #   IBKR Client Portal API wrapper
│   │   ├── ibkr_feed.py                #   Cached feed + circuit breaker
│   │   ├── freshness.py                #   Adaptive freshness bounds
│   │   ├── circuit_breaker.py          #   Three-state circuit breaker
│   │   ├── orchestrator.py             #   DataFeedOrchestrator (parallel)
│   │   ├── ticker_weights.py           #   ES/NQ/SPX agent multipliers
│   │   └── benchmarks.py              #   ES/NQ/SPY/QQQ benchmark metrics
│   │
│   ├── backtesting/                     # Backtesting engine
│   │   ├── engine.py                    #   Main backtest coordinator
│   │   ├── portfolio.py                 #   Portfolio tracking
│   │   ├── controller.py               #   Agent control
│   │   ├── trader.py                    #   Trade execution
│   │   ├── metrics.py                   #   Performance metrics
│   │   ├── valuation.py                 #   Portfolio valuation
│   │   ├── benchmarks.py               #   Benchmark comparison
│   │   ├── output.py                    #   Result formatting
│   │   └── types.py                     #   Type definitions
│   │
│   ├── data/                            # Data models + cache
│   │   ├── models.py                    #   Pydantic models (Price, Metrics, etc.)
│   │   └── cache.py                     #   In-memory cache
│   │
│   ├── tools/
│   │   └── api.py                       #   Financial Datasets API wrapper
│   │
│   ├── graph/
│   │   └── state.py                     #   LangGraph agent state
│   │
│   ├── llm/
│   │   ├── models.py                    #   Multi-provider LLM config
│   │   ├── api_models.json              #   Cloud model list
│   │   └── ollama_models.json           #   Local model list
│   │
│   └── utils/
│       ├── analysts.py                  #   Analyst registry
│       ├── llm.py                       #   LLM calling with retries
│       ├── display.py                   #   CLI output formatting
│       ├── progress.py                  #   Progress tracking
│       ├── ollama.py                    #   Ollama integration
│       └── visualize.py                 #   Graph visualization
│
├── app/
│   ├── backend/
│   │   ├── main.py                      #   FastAPI app entry
│   │   ├── routes/                      #   8 route modules
│   │   │   ├── hedge_fund.py            #     Run + backtest (SSE)
│   │   │   ├── flows.py                 #     Flow CRUD
│   │   │   ├── flow_runs.py             #     Flow run lifecycle
│   │   │   ├── api_keys.py              #     Key management
│   │   │   ├── language_models.py       #     Model listing
│   │   │   ├── ollama.py                #     Local model mgmt
│   │   │   ├── storage.py               #     File save
│   │   │   └── health.py                #     Health checks
│   │   ├── services/                    #   6 service modules
│   │   ├── models/                      #   Pydantic schemas + SSE events
│   │   ├── repositories/               #   3 data access modules
│   │   ├── database/                    #   SQLAlchemy + migrations
│   │   └── alembic/                     #   Migration scripts
│   │
│   └── frontend/
│       ├── package.json                 #   Node dependencies
│       └── src/
│           ├── App.tsx                  #   Root component
│           ├── main.tsx                 #   Entry point
│           ├── components/              #   60+ components
│           │   ├── Layout.tsx           #     VS Code-style IDE layout
│           │   ├── Flow.tsx             #     ReactFlow canvas
│           │   ├── panels/              #     Left/right/bottom panels
│           │   ├── tabs/                #     Tab management
│           │   ├── settings/            #     Settings dialog
│           │   ├── nodes/               #     Custom ReactFlow nodes
│           │   ├── edges/               #     Custom edges
│           │   └── ui/                  #     Shadcn UI components
│           ├── contexts/                #   4 React contexts
│           ├── hooks/                   #   7+ custom hooks
│           ├── services/                #   5 API services
│           └── providers/               #   Theme provider
│
├── tests/
│   ├── strategies/                      #   81 strategy tests
│   ├── feeds/                           #   70 feed tests
│   ├── backtesting/                     #   Engine/portfolio/metrics tests
│   └── fixtures/                        #   Test data
│
└── docker/
    ├── Dockerfile
    └── docker-compose.yml
```

---

## 12. Disclaimer

This project is for **educational and research purposes only**. It is not intended as financial advice or a recommendation to trade. No real money should be risked based on this system's outputs. The creators assume no liability for financial losses.
