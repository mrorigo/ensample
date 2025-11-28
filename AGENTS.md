### **MDAPFlow-MCP Architectural Overview for Component Development Team**

#### **1. The Goal: What is `MDAPFlow-MCP`?**

The **Cognitive Code Engineer (CCE)** is an enterprise-grade AI system designed for highly reliable, multi-step code transformation tasks. The fundamental challenge it addresses is the inherent unpredictability and error rate of Large Language Models (LLMs), which, if unmitigated, renders them unsuitable for critical enterprise workflows.

**`MDAPFlow-MCP` is the dedicated, high-assurance LLM execution engine of this system.** Your team is building the core reliability component that empowers any client to leverage LLMs with unprecedented confidence. It operationalizes the principles of **Massively Decomposed Agentic Processes (MDAPs)**, drawing directly from the research detailed in "Solving a Million-Step LLM Task with Zero Errors."

Instead of clients directly calling LLMs and managing complex retry, validation, and ensemble logic, `MDAPFlow-MCP` centralizes all of this. Your component provides a single, reliable endpoint for LLM inference, transforming probabilistic outputs into high-confidence decisions.

#### **2. The Core Architecture: Your Place in the CCE Ecosystem**

The CCE is now built on a highly modular architecture, where the monolithic `CCE Manager` has been decomposed into specialized internal services and dedicated external MCP servers. `MDAPFlow-MCP` is one of these critical new external MCP servers.

*   **Your Primary Client (within CCE): The `Task Orchestrator Service`**
    *   This is the central "brain" of the CCE application. It manages the entire task lifecycle (planning, execution, verification, summarization).
    *   It defines the *role* an LLM needs to play (e.g., "Planner," "Executor," "Critic") and constructs the specific natural language `prompt` for that role.
    *   It determines the desired reliability level and corresponding MDAP parameters (`voting_k`, `ensemble_config`, `red_flag_config`, `fast_path_enabled`).
    *   The `Task Orchestrator Service` **will be calling `MDAPFlow-MCP` for virtually *every* LLM-driven decision** it needs to make. It delegates the "how to get a reliable LLM response" problem entirely to you.

*   **Other Key CCE Components (Context):**
    *   **`RAILFlow-MCP` (New MCP Server):** Handles all governance (policy evaluation, HIL, auditing). The `Task Orchestrator Service` will often send *your output* (a reliable LLM response) to `RAILFlow-MCP` for policy checking *before* proceeding with the next step.
    *   **`MemoryFlow-MCP` (Cognitive Project Memory):** Stores deep project context. The `Task Orchestrator Service` will query `MemoryFlow-MCP` *before* constructing prompts it sends to you, to ensure context-aware reasoning.
    *   **`CodeFlow-MCP` / `CodeWrite-MCP` / `DocsFlow-MCP` (Expert MCPs):** These are the "hands," "eyes," and "librarian" of the CCE. The `Task Orchestrator Service` will call them (often with outputs from `MDAPFlow-MCP`) to perform physical actions or gather data.

**Your `MDAPFlow-MCP` server is the crucible where raw LLM potential is forged into high-assurance decisions, abstracting that complexity from the rest of the CCE.**

#### **3. The Flow of an LLM Decision: How Your Component Works**

Understanding this flow is critical to designing and implementing `MDAPFlow-MCP`.

1.  **Need for LLM Output:** The `Task Orchestrator Service` (or any external client) determines it needs an LLM to perform a specific function (e.g., generate a code snippet, classify user intent, summarize a report).

2.  **Prompt & Reliability Parameters:** The client constructs the precise natural language `prompt` and defines the `role_name`. Crucially, it also specifies the desired MDAP parameters within the `MDAPInput` object:
    *   `voting_k` (e.g., `3` for high confidence, `0` for greedy/fast-path).
    *   `ensemble_config` (which LLMs to use).
    *   `red_flag_config` (rules for filtering bad outputs).
    *   `output_parser_schema` (how to validate structured outputs).
    *   `fast_path_enabled` (if a quick, potentially lower-assurance path is acceptable).

3.  **Dispatch to You:** The client dispatches a `mdapflow.execute_llm_role` tool call to **`MDAPFlow-MCP`** (your server!).

4.  **MDAP Execution (Your Core Responsibility):**
    *   `MDAPFlow-MCP` receives the `MDAPInput`.
    *   It dispatches **multiple, parallel LLM calls** using its `EnsembleManager` (selecting models from the `ensemble_config`).
    *   Each raw LLM response is then passed through the `RedFlaggingEngine` (discarding unreliable outputs) and the `OutputParser` (canonicalizing structured responses).
    *   The `VotingMechanism` counts valid votes. This process continues in rounds until the "first-to-ahead-by-k" condition is met.
    *   If `fast_path_enabled` is true, the `FastPathController` can short-circuit the voting process early if sufficient confidence is achieved.
    *   All internal LLM interactions (API keys, rate limits, retries, cost estimation) are handled by your `LLMProviderInterface`.

5.  **Reliable Output & Metrics:** `MDAPFlow-MCP` returns a single `MDAPOutput` object, containing:
    *   The `final_response` (the chosen, high-confidence LLM output).
    *   A `confidence_score`.
    *   Detailed `mdap_metrics` (total LLM calls, voting rounds, red flags hit, estimated cost, etc.).

6.  **Client Continues:** The client (e.g., `Task Orchestrator Service`) receives your reliable `final_response` and `mdap_metrics` and proceeds with its next logical step, knowing it can trust the LLM output.

#### **4. Your Role & Boundaries: What `MDAPFlow-MCP` Is (and Is Not)**

*   **You ARE:**
    *   The **sole executor** of all LLM inference operations within the CCE architecture (and any other client).
    *   The **arbiter of LLM reliability**, ensuring that client systems receive high-confidence, validated responses.
    *   Responsible for implementing and optimizing the **full MDAP pipeline** (ensemble management, red-flagging, output parsing, "first-to-ahead-by-k" voting, fast-path).
    *   Responsible for **managing interactions with diverse LLM providers** (API keys, authentication, rate limits, cost estimation).
    *   A **service provider** that receives explicit prompts and reliability configurations, returning a structured LLM response and detailed metrics.
    *   Providing **comprehensive observability** into MDAP execution (tracing, logging, metrics).

*   **You ARE NOT:**
    *   Responsible for **high-level planning or task decomposition** (that's the `Task Orchestrator Service`).
    *   Responsible for **governance policy evaluation or Human-in-the-Loop workflows** (that's `RAILFlow-MCP`).
    *   Responsible for **code analysis, modification, documentation, or cognitive memory** (those are other expert MCPs).
    *   Directly interacting with the **external A2A client**.
    *   Deciding the *content* of the prompt (you execute the prompt you're given), but ensuring the *reliability of the selected output* from that prompt.

In summary, you are building the robust, intelligent reliability layer that underpins the entire CCE's cognitive capabilities. Your component's precision and configurability are paramount to the CCE's ability to operate effectively and trustworthily in enterprise environments. Your focus is on turning LLM statistical probabilities into actionable, high-assurance certainty.


### Project guidelines

- Use `uv` for package management. There is a venv in `.venv` (created with `uv venv`)
