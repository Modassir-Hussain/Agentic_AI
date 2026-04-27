import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast
import json
import re
from dataclasses import asdict, dataclass, field
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from sqlalchemy import create_engine, Engine
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]

########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################


dotenv.load_dotenv()

CATALOG_BY_NAME = {item["item_name"]: item for item in paper_supplies}


@dataclass
class RequestedLineItem:
    """Normalized representation of one requested item extracted from free text."""

    requested_text: str
    quantity: int
    item_name: Optional[str]
    tracked_item: bool
    unit_price: Optional[float]
    match_reason: str


@dataclass
class ParsedCustomerRequest:
    """Structured request passed between the deterministic worker agents."""

    raw_request: str
    request_date: str
    due_date: str
    job: str
    need_size: str
    event: str
    request_kind: str
    items: List[RequestedLineItem] = field(default_factory=list)


@dataclass
class InventoryAssessment:
    """Inventory and supplier feasibility for a single requested item."""

    line_id: int
    requested_text: str
    item_name: Optional[str]
    quantity: int
    tracked_item: bool
    unit_price: Optional[float]
    current_stock: int
    shortage: int
    min_stock_level: int
    projected_remaining_stock: int
    supplier_delivery_date: Optional[str]
    can_meet_due_date: bool
    reorder_recommended: int
    reason: str


@dataclass
class InventoryDecision:
    """Combined inventory review for an entire request."""

    request_date: str
    due_date: str
    cash_balance: float
    total_reorder_cost: float
    can_fulfill: bool
    reasons: List[str]
    item_assessments: List[InventoryAssessment]


@dataclass
class QuoteDecision:
    """Pricing outcome returned by the quoting worker."""

    can_quote: bool
    subtotal: float
    discount_rate: float
    final_total: float
    history_matches: int
    rationale: str


@dataclass
class FulfillmentDecision:
    """Order execution decision and transaction summary."""

    status: str
    fulfilled: bool
    created_transaction_ids: List[int]
    reorder_items: List[Dict[str, Any]]
    reasons: List[str]


@dataclass
class AgentDependencies:
    """Dependencies passed to pydantic-ai agents."""

    system: "MunderDifflinMultiAgentSystem"
    request: ParsedCustomerRequest


ITEM_MATCH_RULES = [
    (r"\b100\s*lb\s*cover\s*stock\b", "100 lb cover stock", "explicit 100 lb cover stock request"),
    (r"\b80\s*lb\s*text\s*paper\b|\b24\s*lb\s*bond\s*paper\b", "80 lb text paper", "text-weight paper request"),
    (r"\bphoto\s*paper\b|\bglossy\s*photo\s*paper\b", "Photo paper", "photo paper wording"),
    (r"\bglossy\b", "Glossy paper", "gloss finish wording"),
    (r"\bmatte\b", "Matte paper", "matte finish wording"),
    (r"\bposter\s*boards?\b|\bposter\s*paper\b|\bposters?\b", "Large poster paper (24x36 inches)", "poster wording"),
    (r"\bbanner\s*paper\b", "Banner paper", "banner wording"),
    (r"\bstreamers?\b", "Crepe paper", "streamer wording mapped to crepe paper"),
    (r"\bwashi\s*tape\b|\bdecorative\s*adhesive\s*tape\b", "Decorative adhesive tape (washi tape)", "washi tape wording"),
    (r"\btable\s*covers?\b", "Table covers", "table cover wording"),
    (r"\bpaper\s*cups?\b|\bdisposable\s*cups?\b|\bbiodegradable\s*cups?\b", "Paper cups", "cup wording"),
    (r"\bpaper\s*plates?\b|\bbiodegradable\s*paper\s*plates?\b", "Paper plates", "plate wording"),
    (r"\bnapkins?\b", "Paper napkins", "napkin wording"),
    (r"\bflyers?\b", "Flyers", "flyer wording"),
    (r"\binvitation\s*cards?\b", "Invitation cards", "invitation card wording"),
    (r"\bpresentation\s*folders?\b", "Presentation folders", "folder wording"),
    (r"\bname\s*tags?\b", "Name tags with lanyards", "name tag wording"),
    (r"\bletterhead\b", "Letterhead paper", "letterhead wording"),
    (r"\blegal\s*size\b", "Legal-size paper", "legal-size wording"),
    (r"\benvelopes?\b", "Envelopes", "envelope wording"),
    (r"\bsticky\s*notes?\b", "Sticky notes", "sticky note wording"),
    (r"\bnotepads?\b", "Notepads", "notepad wording"),
    (r"\bwrapping\s*paper\b", "Wrapping paper", "wrapping paper wording"),
    (r"\bpatterned\s*paper\b", "Patterned paper", "patterned paper wording"),
    (r"\bkraft\s*paper\b", "Kraft paper", "kraft paper wording"),
    (r"\bbutcher\s*paper\b", "Butcher paper", "butcher paper wording"),
    (r"\bconstruction\s*paper\b", "Colored paper", "construction paper mapped to colored paper"),
    (r"\brecycled\s*paper\b", "Recycled paper", "recycled paper wording"),
    (r"\beco\s*friendly\s*paper\b", "Eco-friendly paper", "eco-friendly paper wording"),
    (r"\bcolored\b|\bcolourful\b|\bcolorful\b|\bbright\s*colored\b", "Colored paper", "colored paper wording"),
    (r"\bcard\s*stock\b|\bcardstock\b|\bcover\s*stock\b|\bheavy\s*cardstock\b|\bheavyweight\s*cardstock\b", "Cardstock", "cardstock wording"),
    (r"\ba4\b|\bprinter\s*paper\b|\bprinting\s*paper\b|\bcopy\s*paper\b|\bwhite\s*paper\b|\bplain\s*white\s*printer\s*paper\b", "A4 paper", "general printer paper wording"),
]


def _to_iso_date(value: Union[str, datetime]) -> str:
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    if not value:
        return datetime.now().strftime("%Y-%m-%d")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return value
    return datetime.fromisoformat(value).strftime("%Y-%m-%d")


def _latest_prompt_text(messages: List[ModelMessage]) -> str:
    """Extract the latest text payload passed into a FunctionModel run."""

    if not messages:
        return ""

    latest_message = messages[-1]
    text_parts = []
    for part in getattr(latest_message, "parts", []):
        content = getattr(part, "content", None)
        if isinstance(content, str):
            text_parts.append(content)
    return "\n".join(text_parts)


def _first_user_prompt_text(messages: List[ModelMessage]) -> str:
    """Return the first user prompt text from a model run."""

    for message in messages:
        for part in getattr(message, "parts", []):
            if getattr(part, "part_kind", None) == "user-prompt":
                content = getattr(part, "content", None)
                if isinstance(content, str):
                    return content
    return ""


def _normalize_tool_content(content: Any) -> Any:
    """Convert JSON string tool payloads back into Python data when possible."""

    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return content
    return content


def _tool_returns_by_name(messages: List[ModelMessage]) -> Dict[str, List[Any]]:
    """Collect tool return payloads grouped by tool name."""

    grouped: Dict[str, List[Any]] = {}
    for message in messages:
        for part in getattr(message, "parts", []):
            if getattr(part, "part_kind", None) == "tool-return":
                grouped.setdefault(part.tool_name, []).append(_normalize_tool_content(part.content))
    return grouped


def _json_response(payload: Dict[str, Any]) -> ModelResponse:
    """Return a JSON-encoded model response."""

    return _response_text(json.dumps(payload))


def _response_text(text_value: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(text_value)])


class MunderDifflinMultiAgentSystem:
    """Multi-agent implementation built with pydantic-ai agents."""

    def __init__(self) -> None:
        self.inventory_reference = pd.read_sql("SELECT * FROM inventory", db_engine)
        self.inventory_by_name = {
            row["item_name"]: row.to_dict() for _, row in self.inventory_reference.iterrows()
        }
        self.openai_provider = self._build_openai_provider()
        self.openai_model_name = (
            os.getenv("UDACITY_OPENAI_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        )

        self.inventory_agent = Agent(
            FunctionModel(self._inventory_model),
            deps_type=AgentDependencies,
            instructions=(
                "You are the inventory specialist. Summarize stock levels, reorder needs, "
                "and deadline feasibility clearly."
            ),
        )
        self.quote_agent = Agent(
            FunctionModel(self._quote_model),
            deps_type=AgentDependencies,
            instructions=(
                "You are the quoting specialist. Explain pricing, bulk discounts, and how "
                "historical quotes influenced the offer."
            ),
        )
        self.fulfillment_agent = Agent(
            FunctionModel(self._fulfillment_model),
            deps_type=AgentDependencies,
            instructions=(
                "You are the fulfillment specialist. Confirm which transactions were recorded "
                "and whether delivery is feasible."
            ),
        )
        self.orchestrator_agent = Agent(
            self._build_orchestrator_model(),
            deps_type=AgentDependencies,
            instructions=(
                "You are the customer-facing orchestrator for Munder Difflin. "
                "For order requests, first call delegate_inventory_review and delegate_quote_generation. "
                "After both return, call delegate_fulfillment_check with those exact tool outputs using the argument names inventory_result and quote_result. "
                "Base your final answer only on tool results, never invent stock, pricing, or transaction details. "
                "For inventory-only questions, call answer_inventory_query and use that result. "
                "Keep the final response concise and transparent without exposing internal-only financial detail."
            ),
        )

        self._register_tools()

    def _build_openai_provider(self) -> Optional[OpenAIProvider]:
        """Create an OpenAI-compatible provider from README-style environment variables."""

        api_key = os.getenv("UDACITY_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None

        base_url = (
            os.getenv("UDACITY_OPENAI_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or "https://openai.vocareum.com/v1"
        )
        return OpenAIProvider(base_url=base_url, api_key=api_key)

    def _build_orchestrator_model(self):
        """Use OpenAI for orchestration when configured, otherwise keep the deterministic fallback."""

        if self.openai_provider is None:
            return FunctionModel(self._orchestrator_model)
        return OpenAIChatModel(self.openai_model_name, provider=self.openai_provider)

    def _register_tools(self) -> None:
        @self.orchestrator_agent.tool
        def answer_inventory_query(ctx: RunContext[AgentDependencies]) -> str:
            """Answer a pure inventory question without invoking the order workflow."""

            return ctx.deps.system.handle_inventory_query(ctx.deps.request)

        @self.orchestrator_agent.tool
        def delegate_inventory_review(ctx: RunContext[AgentDependencies]) -> Dict[str, Any]:
            """Delegate inventory feasibility analysis to the inventory worker agent."""

            return ctx.deps.system.delegate_to_inventory_agent(ctx.deps.request)

        @self.orchestrator_agent.tool
        def delegate_quote_generation(ctx: RunContext[AgentDependencies]) -> Dict[str, Any]:
            """Delegate quote generation to the quote worker agent."""

            return ctx.deps.system.delegate_to_quote_agent(ctx.deps.request)

        @self.orchestrator_agent.tool
        def delegate_fulfillment_check(
            ctx: RunContext[AgentDependencies],
            inventory_result: Optional[Dict[str, Any]] = None,
            quote_result: Optional[Dict[str, Any]] = None,
            inventory_review_response: Optional[Dict[str, Any]] = None,
            quote_generation_response: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            """Delegate final transaction handling to the fulfillment worker agent."""

            resolved_inventory = inventory_result or inventory_review_response
            resolved_quote = quote_result or quote_generation_response
            if resolved_inventory is None or resolved_quote is None:
                raise ValueError(
                    "delegate_fulfillment_check requires inventory_result and quote_result tool payloads."
                )

            return ctx.deps.system.delegate_to_fulfillment_agent(
                ctx.deps.request,
                resolved_inventory,
                resolved_quote,
            )

        @self.inventory_agent.tool
        def inventory_snapshot(ctx: RunContext[AgentDependencies], as_of_date: str) -> Dict[str, int]:
            """Return all tracked inventory as of a given ISO date."""

            return ctx.deps.system.tool_inventory_snapshot(as_of_date)

        @self.inventory_agent.tool
        def cash_position(ctx: RunContext[AgentDependencies], as_of_date: str) -> float:
            """Return current cash on hand for reorder feasibility checks."""

            return ctx.deps.system.tool_cash_balance(as_of_date)

        @self.inventory_agent.tool
        def stock_level(
            ctx: RunContext[AgentDependencies], line_id: int, item_name: str, as_of_date: str
        ) -> Dict[str, Union[str, int]]:
            """Return the stock level for one tracked item."""

            result = ctx.deps.system.tool_stock_level(item_name, as_of_date)
            result["line_id"] = line_id
            return result

        @self.inventory_agent.tool
        def supplier_delivery(
            ctx: RunContext[AgentDependencies],
            line_id: int,
            item_name: str,
            as_of_date: str,
            quantity: int,
        ) -> Dict[str, Any]:
            """Estimate the supplier delivery date for a reorder quantity."""

            return {
                "line_id": line_id,
                "item_name": item_name,
                "quantity": quantity,
                "supplier_delivery_date": ctx.deps.system.tool_supplier_delivery(as_of_date, quantity),
            }

        @self.quote_agent.tool
        def quote_history(
            ctx: RunContext[AgentDependencies], search_terms: List[str], limit: int = 5
        ) -> List[Dict[str, Any]]:
            """Search historical quotes related to the current request."""

            return ctx.deps.system.tool_search_quote_history(search_terms, limit=limit)

        @self.quote_agent.tool
        def cash_position(ctx: RunContext[AgentDependencies], as_of_date: str) -> float:
            """Return current cash on hand for internal pricing checks."""

            return ctx.deps.system.tool_cash_balance(as_of_date)

        @self.quote_agent.tool
        def financial_report(ctx: RunContext[AgentDependencies], as_of_date: str) -> Dict[str, Any]:
            """Return a financial snapshot used to keep discounting disciplined."""

            return ctx.deps.system.tool_financial_report(as_of_date)

        @self.fulfillment_agent.tool
        def create_sales_transaction(
            ctx: RunContext[AgentDependencies],
            item_name: str,
            transaction_type: str,
            quantity: int,
            price: float,
            transaction_date: str,
            transaction_label: str,
        ) -> Dict[str, Any]:
            """Record a stock order or sale transaction in the database."""

            transaction_id = ctx.deps.system.tool_create_transaction(
                item_name=item_name,
                transaction_type=transaction_type,
                quantity=quantity,
                price=price,
                transaction_date=transaction_date,
            )
            return {
                "transaction_id": transaction_id,
                "transaction_label": transaction_label,
                "item_name": item_name,
                "transaction_type": transaction_type,
                "quantity": quantity,
                "price": price,
            }

    # Tools for inventory agent
    def tool_inventory_snapshot(self, as_of_date: str) -> Dict[str, int]:
        return get_all_inventory(_to_iso_date(as_of_date))

    def tool_stock_level(self, item_name: str, as_of_date: str) -> Dict[str, Union[str, int]]:
        stock_df = get_stock_level(item_name, _to_iso_date(as_of_date))
        current_stock = 0
        if not stock_df.empty:
            current_stock = int(stock_df["current_stock"].iloc[0])
        return {"item_name": item_name, "current_stock": current_stock}

    def tool_supplier_delivery(self, as_of_date: str, quantity: int) -> str:
        return get_supplier_delivery_date(_to_iso_date(as_of_date), quantity)

    # Tools for quoting agent
    def tool_search_quote_history(self, search_terms: List[str], limit: int = 5) -> List[Dict[str, Any]]:
        filtered_terms = [term for term in search_terms if term]
        return search_quote_history(filtered_terms, limit=limit)

    def tool_cash_balance(self, as_of_date: str) -> float:
        return get_cash_balance(_to_iso_date(as_of_date))

    def tool_financial_report(self, as_of_date: str) -> Dict[str, Any]:
        return generate_financial_report(_to_iso_date(as_of_date))

    # Tools for ordering agent
    def tool_create_transaction(
        self,
        item_name: str,
        transaction_type: str,
        quantity: int,
        price: float,
        transaction_date: str,
    ) -> int:
        return create_transaction(
            item_name=item_name,
            transaction_type=transaction_type,
            quantity=quantity,
            price=price,
            date=_to_iso_date(transaction_date),
        )

    def _parse_due_date(self, text_value: str, fallback_date: str) -> str:
        normalized_text = re.sub(r"(\d{1,2})(st|nd|rd|th)", r"\1", text_value, flags=re.IGNORECASE)
        month_matches = re.findall(
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}",
            normalized_text,
            flags=re.IGNORECASE,
        )
        if month_matches:
            return datetime.strptime(month_matches[-1], "%B %d, %Y").strftime("%Y-%m-%d")

        iso_matches = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", normalized_text)
        if iso_matches:
            return iso_matches[-1]

        return fallback_date

    def _strip_non_item_numbers(self, text_value: str) -> str:
        cleaned = re.sub(r"\(\s*Date of request:[^)]+\)", " ", text_value, flags=re.IGNORECASE)
        cleaned = re.sub(
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}",
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", " ", cleaned)
        cleaned = re.sub(r"\b\d+%", " percentage ", cleaned)
        cleaned = re.sub(r"\d+(?:\.\d+)?\s*(?:\"|inches?)\s*x\s*\d+(?:\.\d+)?\s*(?:\"|inches?)", " size ", cleaned)
        return cleaned.replace("\r", " ")

    def _resolve_catalog_item(self, description: str) -> tuple[Optional[str], str]:
        normalized = re.sub(r"[^a-z0-9]+", " ", description.lower()).strip()

        for pattern, item_name, reason in ITEM_MATCH_RULES:
            if re.search(pattern, normalized):
                return item_name, reason

        return None, "no reliable catalog match found"

    def _extract_items(self, request_text: str) -> List[RequestedLineItem]:
        working_text = self._strip_non_item_numbers(request_text)
        working_text = working_text.replace("\n-", " ").replace("\n•", " ")
        working_text = re.sub(r"\s+", " ", working_text)
        quantity_pattern = re.compile(r"(?<![/x\d])\b\d[\d,]*\b")
        quantity_matches = list(quantity_pattern.finditer(working_text))
        items: List[RequestedLineItem] = []
        for index, match in enumerate(quantity_matches):
            next_start = quantity_matches[index + 1].start() if index + 1 < len(quantity_matches) else len(working_text)
            segment = working_text[match.start():next_start].strip(" ,.;:-")
            segment_match = re.match(r"(?P<qty>\d[\d,]*)\s+(?P<desc>.+)", segment)
            if not segment_match:
                continue

            quantity = int(segment_match.group("qty").replace(",", ""))
            description = segment_match.group("desc")
            description = re.sub(
                r"^(?:sheets?|reams?|rolls?|roll|packets?|packs?|boxes?|pads?|units?)\s*(?:of\s+)?",
                "",
                description,
                flags=re.IGNORECASE,
            )
            description = re.sub(r"^(?:and|along with)\s+", "", description, flags=re.IGNORECASE)
            description = re.split(
                r"\b(?:for\s+our|for\s+the|for\s+an|for\s+upcoming|please|deliver|delivered|delivery|needed|need|must|thank)\b",
                description,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0]
            description = re.sub(r",?\s*(?:and|along with)\s*$", "", description, flags=re.IGNORECASE)
            description = re.sub(r"\b(?:I|We)\s*$", "", description).strip()
            description = description.strip(" ,.;:-")
            if not description:
                continue

            item_name, reason = self._resolve_catalog_item(description)
            tracked_item = item_name in self.inventory_by_name if item_name else False
            unit_price = CATALOG_BY_NAME.get(item_name, {}).get("unit_price") if item_name else None
            items.append(
                RequestedLineItem(
                    requested_text=description,
                    quantity=quantity,
                    item_name=item_name,
                    tracked_item=tracked_item,
                    unit_price=unit_price,
                    match_reason=reason,
                )
            )
        return items

    def parse_customer_request(
        self,
        request_text: str,
        request_date: str,
        job: str,
        need_size: str,
        event: str,
    ) -> ParsedCustomerRequest:
        due_date = self._parse_due_date(request_text, request_date)
        items = self._extract_items(request_text)
        request_kind = "inventory_query" if not items and "inventory" in request_text.lower() else "order_request"
        return ParsedCustomerRequest(
            raw_request=request_text,
            request_date=request_date,
            due_date=due_date,
            job=job,
            need_size=need_size,
            event=event,
            request_kind=request_kind,
            items=items,
        )

    def _parsed_request_from_payload(self, payload: Dict[str, Any]) -> ParsedCustomerRequest:
        return ParsedCustomerRequest(
            raw_request=payload["raw_request"],
            request_date=payload["request_date"],
            due_date=payload["due_date"],
            job=payload["job"],
            need_size=payload["need_size"],
            event=payload["event"],
            request_kind=payload["request_kind"],
            items=[RequestedLineItem(**item) for item in payload["items"]],
        )

    def _request_payload(self, parsed_request: ParsedCustomerRequest) -> Dict[str, Any]:
        """Return a JSON-serializable representation of a parsed request."""

        return asdict(parsed_request)

    def _render_inventory_summary(self, inventory_result: Dict[str, Any]) -> str:
        """Render the inventory worker result into customer-safe text."""

        if inventory_result["can_fulfill"]:
            ready_items = []
            reorder_items = []
            for item in inventory_result["item_assessments"]:
                if item["shortage"]:
                    reorder_items.append(
                        f"{item['item_name']} needs {item['shortage']} more units and can arrive by {item['supplier_delivery_date']}"
                    )
                else:
                    ready_items.append(f"{item['item_name']} ({item['current_stock']} on hand)")

            summary_parts = []
            if ready_items:
                summary_parts.append("In stock now: " + ", ".join(ready_items) + ".")
            if reorder_items:
                summary_parts.append("Supplier action: " + "; ".join(reorder_items) + ".")
            summary_parts.append(
                f"The full order can still meet the requested delivery date of {inventory_result['due_date']}."
            )
            return " ".join(summary_parts)

        blocked_items = "; ".join(inventory_result["reasons"]) if inventory_result["reasons"] else "Inventory constraints block this request."
        return f"Inventory review found blockers: {blocked_items}"

    def _build_inventory_result(
        self, parsed_request: ParsedCustomerRequest, tool_returns: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """Assemble the inventory decision from worker tool returns."""

        inventory_snapshot = tool_returns.get("inventory_snapshot", [{}])[-1]
        if not isinstance(inventory_snapshot, dict):
            inventory_snapshot = {}
        cash_balance = float(tool_returns.get("cash_position", [0.0])[-1] or 0.0)
        stock_level_returns = {
            int(item["line_id"]): item
            for item in tool_returns.get("stock_level", [])
            if isinstance(item, dict) and "line_id" in item
        }
        supplier_returns = {
            int(item["line_id"]): item
            for item in tool_returns.get("supplier_delivery", [])
            if isinstance(item, dict) and "line_id" in item
        }

        item_assessments: List[InventoryAssessment] = []
        reasons: List[str] = []
        for line_id, item in enumerate(parsed_request.items):
            if not item.item_name:
                assessment = InventoryAssessment(
                    line_id=line_id,
                    requested_text=item.requested_text,
                    item_name=None,
                    quantity=item.quantity,
                    tracked_item=False,
                    unit_price=None,
                    current_stock=0,
                    shortage=item.quantity,
                    min_stock_level=0,
                    projected_remaining_stock=0,
                    supplier_delivery_date=None,
                    can_meet_due_date=False,
                    reorder_recommended=0,
                    reason=f"Could not map '{item.requested_text}' to a supported catalog item.",
                )
                reasons.append(assessment.reason)
                item_assessments.append(assessment)
                continue

            if not item.tracked_item:
                assessment = InventoryAssessment(
                    line_id=line_id,
                    requested_text=item.requested_text,
                    item_name=item.item_name,
                    quantity=item.quantity,
                    tracked_item=False,
                    unit_price=item.unit_price,
                    current_stock=0,
                    shortage=item.quantity,
                    min_stock_level=0,
                    projected_remaining_stock=0,
                    supplier_delivery_date=None,
                    can_meet_due_date=False,
                    reorder_recommended=0,
                    reason=(
                        f"{item.item_name} exists in the broader catalog but is not in the tracked inventory subset "
                        "seeded for this simulation."
                    ),
                )
                reasons.append(assessment.reason)
                item_assessments.append(assessment)
                continue

            current_stock = int(
                stock_level_returns.get(line_id, {}).get("current_stock", inventory_snapshot.get(item.item_name, 0))
            )
            shortage = max(0, item.quantity - current_stock)
            inventory_row = self.inventory_by_name[item.item_name]
            min_stock_level = int(inventory_row["min_stock_level"])
            supplier_delivery_date = None
            can_meet_due_date = True
            if shortage:
                supplier_delivery_date = supplier_returns.get(line_id, {}).get("supplier_delivery_date")
                can_meet_due_date = bool(supplier_delivery_date) and supplier_delivery_date <= parsed_request.due_date
            projected_remaining_stock = current_stock + shortage - item.quantity
            reorder_recommended = max(0, min_stock_level - projected_remaining_stock)

            if shortage and not can_meet_due_date:
                reason = (
                    f"{item.item_name} needs {shortage} more units and the supplier estimate of "
                    f"{supplier_delivery_date} misses the requested delivery date."
                )
                reasons.append(reason)
            elif shortage:
                reason = (
                    f"{item.item_name} needs a {shortage}-unit supplier order, which can arrive by "
                    f"{supplier_delivery_date}."
                )
            else:
                reason = f"{item.item_name} is already in stock."

            item_assessments.append(
                InventoryAssessment(
                    line_id=line_id,
                    requested_text=item.requested_text,
                    item_name=item.item_name,
                    quantity=item.quantity,
                    tracked_item=True,
                    unit_price=item.unit_price,
                    current_stock=current_stock,
                    shortage=shortage,
                    min_stock_level=min_stock_level,
                    projected_remaining_stock=projected_remaining_stock,
                    supplier_delivery_date=supplier_delivery_date,
                    can_meet_due_date=can_meet_due_date,
                    reorder_recommended=reorder_recommended,
                    reason=reason,
                )
            )

        total_reorder_cost = round(
            sum(
                assessment.shortage * float(assessment.unit_price)
                for assessment in item_assessments
                if assessment.unit_price is not None
            ),
            2,
        )
        if total_reorder_cost > cash_balance:
            reasons.append(
                f"Reordering requires ${total_reorder_cost:.2f}, which exceeds the available cash for the request date."
            )

        inventory_result = {
            "request_date": parsed_request.request_date,
            "due_date": parsed_request.due_date,
            "cash_balance": round(cash_balance, 2),
            "total_reorder_cost": total_reorder_cost,
            "can_fulfill": not reasons and total_reorder_cost <= cash_balance,
            "reasons": reasons,
            "item_assessments": [asdict(item) for item in item_assessments],
        }
        inventory_result["summary"] = self._render_inventory_summary(inventory_result)
        return inventory_result

    def _render_quote_summary(self, quote_result: Dict[str, Any]) -> str:
        """Render the quote worker result into customer-safe text."""

        if not quote_result["can_quote"]:
            return quote_result["rationale"]
        return (
            f"Quote prepared at ${quote_result['final_total']:.2f} from a ${quote_result['subtotal']:.2f} subtotal with a "
            f"{quote_result['discount_rate']:.0%} discount. {quote_result['rationale']}"
        )

    def _build_quote_result(
        self, parsed_request: ParsedCustomerRequest, tool_returns: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """Assemble the quote decision from worker tool returns."""

        recognized_items = [item for item in parsed_request.items if item.item_name and item.unit_price is not None]
        if len(recognized_items) != len(parsed_request.items):
            quote_result = {
                "can_quote": False,
                "subtotal": 0.0,
                "discount_rate": 0.0,
                "final_total": 0.0,
                "history_matches": 0,
                "rationale": "A reliable quote could not be generated because one or more requested items were not mapped.",
            }
            quote_result["summary"] = self._render_quote_summary(quote_result)
            return quote_result

        financial_report = tool_returns.get("financial_report", [{}])[-1]
        if not isinstance(financial_report, dict):
            financial_report = {}
        history_matches = tool_returns.get("quote_history", [[]])[-1]
        if not isinstance(history_matches, list):
            history_matches = []

        subtotal = round(sum(item.quantity * float(item.unit_price) for item in recognized_items), 2)
        total_units = sum(item.quantity for item in recognized_items)
        discount_rate = 0.0
        if total_units >= 1500:
            discount_rate += 0.15
        elif total_units >= 800:
            discount_rate += 0.10
        elif total_units >= 300:
            discount_rate += 0.05

        if parsed_request.need_size == "large":
            discount_rate += 0.03
        elif parsed_request.need_size == "medium":
            discount_rate += 0.01

        if len(history_matches) >= 3:
            discount_rate += 0.02

        if float(financial_report.get("cash_balance", 0.0)) < 10000:
            discount_rate = max(0.0, discount_rate - 0.03)

        discount_rate = min(discount_rate, 0.18)
        final_total = round(subtotal * (1 - discount_rate), 2)
        quote_result = {
            "can_quote": True,
            "subtotal": subtotal,
            "discount_rate": discount_rate,
            "final_total": final_total,
            "history_matches": len(history_matches),
            "rationale": (
                f"Base subtotal is ${subtotal:.2f}. Applied a {discount_rate:.0%} bulk discount using order size "
                f"and {len(history_matches)} related historical quote matches as guidance."
            ),
        }
        quote_result["summary"] = self._render_quote_summary(quote_result)
        return quote_result

    def _allocate_sales_prices(
        self, inventory_result: Dict[str, Any], quote_result: Dict[str, Any]
    ) -> Dict[int, float]:
        """Allocate the final quoted revenue across item lines."""

        subtotals = [
            (
                int(assessment["line_id"]),
                round(assessment["quantity"] * float(assessment["unit_price"]), 2),
            )
            for assessment in inventory_result["item_assessments"]
            if assessment["item_name"] and assessment["unit_price"] is not None
        ]
        if not subtotals:
            return {}

        allocated: Dict[int, float] = {}
        running_total = 0.0
        for index, (line_id, line_subtotal) in enumerate(subtotals):
            if index == len(subtotals) - 1:
                line_total = round(quote_result["final_total"] - running_total, 2)
            else:
                ratio = line_subtotal / quote_result["subtotal"] if quote_result["subtotal"] else 0
                line_total = round(quote_result["final_total"] * ratio, 2)
                running_total += line_total
            allocated[line_id] = line_total
        return allocated

    def _planned_fulfillment_actions(
        self,
        parsed_request: ParsedCustomerRequest,
        inventory_result: Dict[str, Any],
        quote_result: Dict[str, Any],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Build the fulfillment tool-call plan for confirmed orders."""

        if not inventory_result["can_fulfill"] or not quote_result["can_quote"]:
            return [], []

        actions: List[Dict[str, Any]] = []
        reorder_items: List[Dict[str, Any]] = []
        sales_prices = self._allocate_sales_prices(inventory_result, quote_result)

        for assessment in inventory_result["item_assessments"]:
            if assessment["shortage"] > 0 and assessment["item_name"]:
                reorder_cost = round(assessment["shortage"] * float(assessment["unit_price"]), 2)
                label = f"reorder:{assessment['line_id']}"
                actions.append(
                    {
                        "item_name": assessment["item_name"],
                        "transaction_type": "stock_orders",
                        "quantity": assessment["shortage"],
                        "price": reorder_cost,
                        "transaction_date": parsed_request.request_date,
                        "transaction_label": label,
                    }
                )
                reorder_items.append(
                    {
                        "item_name": assessment["item_name"],
                        "quantity": assessment["shortage"],
                        "expected_delivery": assessment["supplier_delivery_date"],
                    }
                )

        for assessment in inventory_result["item_assessments"]:
            if assessment["item_name"]:
                line_id = int(assessment["line_id"])
                actions.append(
                    {
                        "item_name": assessment["item_name"],
                        "transaction_type": "sales",
                        "quantity": assessment["quantity"],
                        "price": float(sales_prices.get(line_id, 0.0)),
                        "transaction_date": parsed_request.request_date,
                        "transaction_label": f"sale:{line_id}",
                    }
                )

        return actions, reorder_items

    def _render_fulfillment_summary(self, fulfillment_result: Dict[str, Any]) -> str:
        """Render the fulfillment worker result into customer-safe text."""

        if not fulfillment_result["fulfilled"]:
            reason_text = "; ".join(fulfillment_result["reasons"]) if fulfillment_result["reasons"] else "The order was not fulfilled."
            return f"Fulfillment blocked. {reason_text}"

        if fulfillment_result["reorder_items"]:
            reorder_text = ", ".join(
                f"{item['quantity']} units of {item['item_name']}" for item in fulfillment_result["reorder_items"]
            )
            return (
                "Order confirmed and transactions were recorded. Additional supplier orders were placed for "
                f"{reorder_text}."
            )

        return "Order confirmed and all required transactions were recorded from current stock."

    def _build_fulfillment_result(
        self,
        parsed_request: ParsedCustomerRequest,
        inventory_result: Dict[str, Any],
        quote_result: Dict[str, Any],
        tool_returns: Dict[str, List[Any]],
    ) -> Dict[str, Any]:
        """Assemble the fulfillment decision from worker tool returns."""

        if not inventory_result["can_fulfill"]:
            fulfillment_result = {
                "status": "rejected",
                "fulfilled": False,
                "created_transaction_ids": [],
                "reorder_items": [],
                "reasons": inventory_result["reasons"],
            }
            fulfillment_result["summary"] = self._render_fulfillment_summary(fulfillment_result)
            return fulfillment_result

        if not quote_result["can_quote"]:
            fulfillment_result = {
                "status": "rejected",
                "fulfilled": False,
                "created_transaction_ids": [],
                "reorder_items": [],
                "reasons": [quote_result["rationale"]],
            }
            fulfillment_result["summary"] = self._render_fulfillment_summary(fulfillment_result)
            return fulfillment_result

        actions, reorder_items = self._planned_fulfillment_actions(parsed_request, inventory_result, quote_result)
        transaction_returns = {
            item["transaction_label"]: item
            for item in tool_returns.get("create_sales_transaction", [])
            if isinstance(item, dict) and "transaction_label" in item
        }
        created_transaction_ids = [
            transaction_returns[action["transaction_label"]]["transaction_id"]
            for action in actions
            if action["transaction_label"] in transaction_returns
        ]
        fulfillment_result = {
            "status": "confirmed",
            "fulfilled": True,
            "created_transaction_ids": created_transaction_ids,
            "reorder_items": reorder_items,
            "reasons": ["Order recorded successfully."],
        }
        fulfillment_result["summary"] = self._render_fulfillment_summary(fulfillment_result)
        return fulfillment_result

    def handle_inventory_query(self, parsed_request: ParsedCustomerRequest) -> str:
        snapshot = self.tool_inventory_snapshot(parsed_request.request_date)
        if not snapshot:
            return "I do not currently have any tracked stock available in the system snapshot."
        top_items = sorted(snapshot.items(), key=lambda pair: (-pair[1], pair[0]))[:8]
        rendered_items = ", ".join(f"{item_name}: {stock}" for item_name, stock in top_items)
        return f"Current tracked inventory as of {parsed_request.request_date}: {rendered_items}."

    def delegate_to_inventory_agent(self, parsed_request: ParsedCustomerRequest) -> Dict[str, Any]:
        """Run the inventory worker agent and return its structured output."""

        deps = AgentDependencies(system=self, request=parsed_request)
        result_text = self.inventory_agent.run_sync(
            json.dumps(self._request_payload(parsed_request)),
            deps=deps,
        ).output
        return json.loads(result_text)

    def delegate_to_quote_agent(self, parsed_request: ParsedCustomerRequest) -> Dict[str, Any]:
        """Run the quote worker agent and return its structured output."""

        deps = AgentDependencies(system=self, request=parsed_request)
        result_text = self.quote_agent.run_sync(
            json.dumps(self._request_payload(parsed_request)),
            deps=deps,
        ).output
        return json.loads(result_text)

    def delegate_to_fulfillment_agent(
        self,
        parsed_request: ParsedCustomerRequest,
        inventory_result: Dict[str, Any],
        quote_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run the fulfillment worker agent and return its structured output."""

        deps = AgentDependencies(system=self, request=parsed_request)
        result_text = self.fulfillment_agent.run_sync(
            json.dumps(
                {
                    "request": self._request_payload(parsed_request),
                    "inventory_result": inventory_result,
                    "quote_result": quote_result,
                }
            ),
            deps=deps,
        ).output
        return json.loads(result_text)

    def handle_customer_request(
        self,
        request_text: str,
        request_date: str,
        job: str,
        need_size: str,
        event: str,
    ) -> str:
        parsed_request = self.parse_customer_request(
            request_text=request_text,
            request_date=request_date,
            job=job,
            need_size=need_size,
            event=event,
        )
        deps = AgentDependencies(system=self, request=parsed_request)
        orchestrator_payload = {
            "raw_request": parsed_request.raw_request,
            "request_date": parsed_request.request_date,
            "due_date": parsed_request.due_date,
            "job": parsed_request.job,
            "need_size": parsed_request.need_size,
            "event": parsed_request.event,
            "request_kind": parsed_request.request_kind,
            "items": [asdict(item) for item in parsed_request.items],
        }
        return self.orchestrator_agent.run_sync(json.dumps(orchestrator_payload), deps=deps).output

    def _inventory_model(self, messages: List[ModelMessage], info: AgentInfo) -> ModelResponse:
        payload = json.loads(_first_user_prompt_text(messages))
        parsed_request = self._parsed_request_from_payload(payload)
        tool_returns = _tool_returns_by_name(messages)

        if not tool_returns:
            tool_calls = [
                ToolCallPart("inventory_snapshot", {"as_of_date": parsed_request.request_date}),
                ToolCallPart("cash_position", {"as_of_date": parsed_request.request_date}),
            ]
            for line_id, item in enumerate(parsed_request.items):
                if item.item_name and item.tracked_item:
                    tool_calls.append(
                        ToolCallPart(
                            "stock_level",
                            {
                                "line_id": line_id,
                                "item_name": item.item_name,
                                "as_of_date": parsed_request.request_date,
                            },
                        )
                    )
            return ModelResponse(parts=tool_calls)

        supplier_return_ids = {
            int(item["line_id"])
            for item in tool_returns.get("supplier_delivery", [])
            if isinstance(item, dict) and "line_id" in item
        }
        stock_returns = {
            int(item["line_id"]): item
            for item in tool_returns.get("stock_level", [])
            if isinstance(item, dict) and "line_id" in item
        }
        supplier_calls = []
        for line_id, item in enumerate(parsed_request.items):
            if not item.item_name or not item.tracked_item:
                continue
            stock_result = stock_returns.get(line_id)
            if not stock_result:
                continue
            shortage = max(0, item.quantity - int(stock_result["current_stock"]))
            if shortage > 0 and line_id not in supplier_return_ids:
                supplier_calls.append(
                    ToolCallPart(
                        "supplier_delivery",
                        {
                            "line_id": line_id,
                            "item_name": item.item_name,
                            "as_of_date": parsed_request.request_date,
                            "quantity": shortage,
                        },
                    )
                )

        if supplier_calls:
            return ModelResponse(parts=supplier_calls)

        return _json_response(self._build_inventory_result(parsed_request, tool_returns))

    def _quote_model(self, messages: List[ModelMessage], info: AgentInfo) -> ModelResponse:
        payload = json.loads(_first_user_prompt_text(messages))
        parsed_request = self._parsed_request_from_payload(payload)
        tool_returns = _tool_returns_by_name(messages)

        if not tool_returns:
            search_terms = [parsed_request.job, parsed_request.event]
            search_terms.extend(item.item_name.split()[0] for item in parsed_request.items if item.item_name)
            return ModelResponse(
                parts=[
                    ToolCallPart("quote_history", {"search_terms": search_terms, "limit": 5}),
                    ToolCallPart("financial_report", {"as_of_date": parsed_request.request_date}),
                ]
            )

        return _json_response(self._build_quote_result(parsed_request, tool_returns))

    def _fulfillment_model(self, messages: List[ModelMessage], info: AgentInfo) -> ModelResponse:
        payload = json.loads(_first_user_prompt_text(messages))
        parsed_request = self._parsed_request_from_payload(payload["request"])
        inventory_result = payload["inventory_result"]
        quote_result = payload["quote_result"]
        tool_returns = _tool_returns_by_name(messages)

        actions, _ = self._planned_fulfillment_actions(parsed_request, inventory_result, quote_result)
        if inventory_result["can_fulfill"] and quote_result["can_quote"] and not tool_returns and actions:
            return ModelResponse(
                parts=[ToolCallPart("create_sales_transaction", action) for action in actions]
            )

        return _json_response(
            self._build_fulfillment_result(parsed_request, inventory_result, quote_result, tool_returns)
        )

    def _orchestrator_model(self, messages: List[ModelMessage], info: AgentInfo) -> ModelResponse:
        payload = json.loads(_first_user_prompt_text(messages))
        parsed_request = self._parsed_request_from_payload(payload)

        if parsed_request.request_kind == "inventory_query":
            return _response_text(self.handle_inventory_query(parsed_request))

        tool_returns = _tool_returns_by_name(messages)
        inventory_results = tool_returns.get("delegate_inventory_review", [])
        quote_results = tool_returns.get("delegate_quote_generation", [])
        fulfillment_results = tool_returns.get("delegate_fulfillment_check", [])

        if not inventory_results and not quote_results:
            return ModelResponse(
                parts=[
                    ToolCallPart("delegate_inventory_review", {}),
                    ToolCallPart("delegate_quote_generation", {}),
                ]
            )

        inventory_result = inventory_results[-1] if inventory_results else {
            "can_fulfill": False,
            "reasons": ["Inventory review did not complete."],
            "summary": "Inventory review did not complete.",
        }
        quote_result = quote_results[-1] if quote_results else {
            "can_quote": False,
            "final_total": 0.0,
            "discount_rate": 0.0,
            "rationale": "Quote generation did not complete.",
            "summary": "Quote generation did not complete.",
        }

        if not fulfillment_results:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "delegate_fulfillment_check",
                        {
                            "inventory_result": inventory_result,
                            "quote_result": quote_result,
                        },
                    )
                ]
            )

        fulfillment_result = fulfillment_results[-1]
        inventory_summary = inventory_result["summary"]
        quote_summary = quote_result["summary"]
        fulfillment_summary = fulfillment_result["summary"]

        rendered_items = []
        for item in payload["items"]:
            label = item["item_name"] or item["requested_text"]
            rendered_items.append(f"{item['quantity']} of {label}")
        item_text = ", ".join(rendered_items)

        if fulfillment_result["fulfilled"]:
            response = (
                f"Order confirmed for {payload['job']} support at the {payload['event']}. "
                f"We can deliver {item_text} by {payload['due_date']}. "
                f"Your quote is ${quote_result['final_total']:.2f}, including a {quote_result['discount_rate']:.0%} bulk discount. "
                f"{inventory_summary} {quote_summary} {fulfillment_summary}"
            )
            return _response_text(response)

        reason_text = "; ".join(fulfillment_result["reasons"] or inventory_result["reasons"])
        if not reason_text:
            reason_text = "The request could not be confirmed."
        response = (
            f"I cannot confirm this order for delivery by {payload['due_date']}. "
            f"Requested items were {item_text}. "
            f"Reason: {reason_text}. "
            f"{inventory_summary} {quote_summary} {fulfillment_summary}"
        )
        return _response_text(response)


# Run your test scenarios by writing them here. Make sure to keep track of them.

def run_test_scenarios():
    
    print("Initializing Database...")
    init_database(db_engine)
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    ############
    ############
    ############
    # INITIALIZE YOUR MULTI AGENT SYSTEM HERE
    ############
    ############
    ############
    multi_agent_system = MunderDifflinMultiAgentSystem()

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        request_with_date = f"{row['request']} (Date of request: {request_date})"

        ############
        ############
        ############
        # USE YOUR MULTI AGENT SYSTEM TO HANDLE THE REQUEST
        ############
        ############
        ############
        response = multi_agent_system.handle_customer_request(
            request_text=request_with_date,
            request_date=request_date,
            job=row["job"],
            need_size=row["need_size"],
            event=row["event"],
        )

        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
            }
        )

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results


if __name__ == "__main__":
    results = run_test_scenarios()
