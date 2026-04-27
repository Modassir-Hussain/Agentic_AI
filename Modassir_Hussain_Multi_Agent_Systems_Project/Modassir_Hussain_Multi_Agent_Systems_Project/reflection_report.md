# Reflection Report

## System Overview

This implementation uses `pydantic-ai` with four agents: an orchestrator, an inventory agent, a quote agent, and a fulfillment agent. I chose `pydantic-ai` because it supports explicit tool registration, OpenAI-compatible hosted models, and deterministic tool-driven workers in the same framework. In the current version, the orchestrator uses the OpenAI-compatible endpoint described in the README when `UDACITY_OPENAI_API_KEY` is present, while the worker agents remain deterministic so inventory, pricing, and transaction decisions stay reproducible and grounded in the starter helper functions.

The orchestrator is now the single entrypoint for nontrivial requests. `handle_customer_request()` only parses the request and hands control to the orchestrator agent. Inside the framework loop, the orchestrator first emits delegation tool calls for the inventory worker and quote worker. Those worker agents then run their own tool-calling steps: the inventory agent calls inventory snapshot, stock-level, supplier ETA, and cash-position tools, while the quote agent calls quote-history and financial-report tools. After those worker results return through the framework, the orchestrator emits a fulfillment delegation tool call. The fulfillment agent is the only component that changes the database, and it does so through transaction-tool calls for stock orders and sales. This keeps delegation, worker execution, and database mutation inside the agent framework rather than in manual Python flow control.

## Evaluation Results

The system was evaluated with the full `quote_requests_sample.csv` dataset and wrote the outputs to `test_results.csv`. The final run produced 20 evaluated requests, 4 fulfilled orders, and 4 request rows with a cash-balance change. The fulfilled requests were small-to-medium orders that matched the seeded inventory subset and had supplier lead times short enough to preserve the requested deadline. The final recorded business state was a cash balance of `$45,306.87` and an inventory value of `$4,660.05`.

The strongest behavior in the evaluation was conservative order acceptance. The system did not silently accept requests that contained unsupported items such as balloons, tickets, cardboard signage, or catalog items outside the seeded tracked subset. It also rejected orders when supplier lead times missed the customer deadline, which is visible in several April 10 and April 15 requests. On the fulfilled requests, the responses include the committed delivery date, the quote total, the applied bulk discount, and a short explanation of whether the order was handled from stock or with a just-in-time supplier reorder.

## Strengths

- The system satisfies the rubric requirement that not all requests are fulfilled, and the rejection reasons are explicit rather than implied only by missing transactions.
- Customer-facing responses are transparent but do not reveal internal cash balances, profit targets, or raw system errors.
- The orchestration path is now genuinely framework-driven: the orchestrator emits delegation tool calls, worker agents run their own tool loops, and the final response is assembled only after tool-return messages come back through the agent runtime.
- All required starter helper functions are used through agent tools or supporting logic: `create_transaction`, `get_all_inventory`, `get_stock_level`, `get_supplier_delivery_date`, `get_cash_balance`, `generate_financial_report`, and `search_quote_history`.
- The OpenAI-backed orchestrator keeps the top-level flow aligned with the README setup, while the deterministic worker path keeps inventory, pricing, and fulfillment behavior easier to grade and debug than a fully stochastic workflow.

## Areas For Improvement

The current parser uses rule-based extraction and alias matching, which works on the sample dataset but is still brittle on unusual phrasing. The tracked-inventory restriction also means some otherwise valid catalog items are rejected because the starter financial report only values items present in the seeded `inventory` table. That is a deliberate safety choice for this project, but it limits fulfillment coverage.

## Suggested Next Improvements

1. Add a richer normalization layer for customer requests so the system can understand more size variants and synonyms without hard-coded rules.
2. Extend the inventory subsystem so newly ordered catalog items can be added to the tracked inventory table safely, allowing more orders to be fulfilled while keeping financial reporting accurate.
3. Extend the OpenAI-backed approach to the worker agents only if needed for richer conversational reasoning, while preserving the current tool-grounded safeguards around stock, pricing, and database updates.