# Agent Workflow Diagram

```mermaid
flowchart TD
    C[Customer Request\nText order or inventory question]
    O[Orchestrator Agent\nResponsibility: OpenAI-backed coordinator that issues delegation tool calls, receives worker results, and assembles the customer-safe reply]
    I[Inventory Agent\nResponsibility: call stock tools, assess shortages, decide if reorder can meet deadline]
    Q[Quote Agent\nResponsibility: call pricing tools, price the request, apply bulk discounts]
    F[Fulfillment Agent\nResponsibility: call transaction tools, record accepted orders, return final execution status]
    D[(SQLite Database)]
    R[test_results.csv\nEvaluation output]

    IT1[Inventory Snapshot Tool\nPurpose: stock snapshot by item\nHelper: get_all_inventory]
    IT2[Item Stock Tool\nPurpose: per-item stock lookup\nHelper: get_stock_level]
    IT3[Supplier ETA Tool\nPurpose: reorder delivery estimate\nHelper: get_supplier_delivery_date]

    QT1[Quote History Tool\nPurpose: retrieve similar historical quotes\nHelper: search_quote_history]
    QT2[Cash Position Tool\nPurpose: confirm reorder affordability\nHelper: get_cash_balance]
    QT3[Financial Report Tool\nPurpose: keep discounts aligned with business health\nHelper: generate_financial_report]

    FT1[Transaction Tool\nPurpose: persist stock orders and sales\nHelper: create_transaction]

    C --> O
    O --> I
    O --> Q
    O --> F
    F --> O
    O --> C

    I --> IT1 --> D
    I --> IT2 --> D
    I --> IT3

    Q --> QT1 --> D
    Q --> QT2 --> D
    Q --> QT3 --> D

    F --> FT1 --> D
    O --> R
```

## Flow Summary

1. The orchestrator agent is the single request entrypoint and receives the request text plus metadata such as job, event, and request date. When `UDACITY_OPENAI_API_KEY` is configured, this step is handled by the OpenAI-compatible model path described in the README.
2. On its first framework step, the orchestrator emits delegation tool calls for the inventory agent and quote agent instead of running worker logic directly in Python.
3. The inventory agent performs its own multi-step tool loop by calling inventory snapshot, stock level, supplier ETA, and cash-position tools before returning a structured inventory decision.
4. The quote agent performs its own tool loop by calling quote-history and financial-report tools before returning a structured quote decision.
5. After those worker results come back as tool returns, the orchestrator emits a fulfillment delegation tool call. The fulfillment agent then calls the transaction tool for accepted orders and returns a structured execution result.
6. The orchestrator receives the final worker outputs and converts them into a customer-facing response that includes the decision, total quote when available, delivery commitment, and a short rationale.

## Agent Boundaries

- The orchestrator is the only public entrypoint for order requests and owns the framework-level delegation sequence plus final response assembly.
- The orchestrator can run on the OpenAI-compatible endpoint from the README, while the worker agents remain tool-grounded and deterministic.
- The inventory agent owns stock, shortage, and delivery-feasibility decisions.
- The quote agent owns pricing, quote-history use, and discount logic.
- The fulfillment agent owns database-changing actions and transaction recording.