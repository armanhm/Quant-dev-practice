# QuantFlow Phase 3B Part 2: LLM Assistant, Dashboard, Paper Trading, Docker

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an LLM-powered research assistant (Claude API), a Streamlit dashboard, Alpaca paper trading, and Docker containerization -- completing the full QuantFlow platform.

**Architecture:** Four independent features. LLM assistant uses Anthropic SDK with tool-use to call platform functions. Streamlit dashboard uses session_state for persistence. Paper trading uses Alpaca REST API to fetch recent bars and feeds them through the existing engine. Docker wraps everything.

**Tech Stack:** anthropic SDK, streamlit, alpaca-py, Docker

---

## Feature 1: LLM Assistant (Tasks 1-3)

### Task 1: LLM Provider + Tools

Create `quantflow/assistant/` package with provider adapter and tool definitions.

**Files:**
- `quantflow/assistant/__init__.py`
- `quantflow/assistant/provider.py` -- LLMProvider protocol + ClaudeProvider
- `quantflow/assistant/tools.py` -- Tool definitions + execution functions
- `tests/test_assistant/__init__.py`
- `tests/test_assistant/test_tools.py`

### Task 2: Chat Interface + CLI Command

- `quantflow/assistant/chat.py` -- Interactive chat loop with conversation memory
- Add `quantflow chat` command to CLI
- `tests/test_assistant/test_chat.py`

### Task 3: LLM Dependencies

- Add `[project.optional-dependencies] llm = ["anthropic>=0.40"]`

---

## Feature 2: Streamlit Dashboard (Tasks 4-5)

### Task 4: Dashboard App

- `quantflow/dashboard/__init__.py`
- `quantflow/dashboard/app.py` -- Main Streamlit app with pages
- Pages: Data Explorer, Strategy Lab, Results Viewer
- Add `[project.optional-dependencies] dashboard = ["streamlit>=1.30", "plotly>=5.0"]`

### Task 5: Dashboard CLI + Chat Panel

- Add `quantflow dashboard` CLI command
- Chat panel in dashboard (uses LLM assistant if available)

---

## Feature 3: Paper Trading (Tasks 6-7)

### Task 6: Alpaca Paper Trader

- `quantflow/live/__init__.py`
- `quantflow/live/paper_trader.py` -- Fetches bars from Alpaca, feeds through engine
- `tests/test_live/__init__.py`
- `tests/test_live/test_paper_trader.py`
- Add `[project.optional-dependencies] live = ["alpaca-py>=0.30"]`

### Task 7: Paper Trading CLI

- Add `quantflow paper` CLI command

---

## Feature 4: Docker (Task 8)

### Task 8: Containerization

- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`
