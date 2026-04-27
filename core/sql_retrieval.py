"""
Text-to-SQL Hybrid Retrieval.

Most enterprise RAG deployments have both:
  - Unstructured data (documents, PDFs) → vector search
  - Structured data (databases, CSV exports) → SQL

This module handles the SQL path. The router decides which path to use;
this module handles the execution once SQL is chosen.

Pipeline:
  1. Load database schema (tables, columns, types)
  2. LLM generates SQL from natural language + schema
  3. Execute SQL safely (read-only, parameterized)
  4. Format results as RAG-compatible context
  5. Return alongside or instead of vector results

Security:
  - Only SELECT statements are allowed (no DDL/DML)
  - Query timeout enforced
  - Results truncated to prevent context overflow

Usage:
  # In .env:
  SQL_DATABASE_URL=sqlite:///./data/company.db
  # Or: postgresql://user:pass@localhost/dbname

  # Create tables + load data:
  python -c "from core.sql_retrieval import create_sample_db; create_sample_db()"

  # Query:
  result = query_natural_language("Top 5 products by revenue")
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default SQLite database path
DEFAULT_DB_PATH = Path("./data/rag_structured.db")


def get_db_url(database: str | None = None) -> str:
    """Resolve database URL from argument or config."""
    if database:
        if database.startswith("sqlite") or database.startswith("postgresql") or database.startswith("mysql"):
            return database
        return f"sqlite:///{database}"

    try:
        from config import settings
        url = getattr(settings, "sql_database_url", "")
        if url:
            return url
    except Exception:
        pass

    DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{DEFAULT_DB_PATH}"


def get_schema(database: str | None = None) -> str:
    """
    Extract the database schema as a CREATE TABLE string.

    The schema is passed to the LLM to enable accurate SQL generation.
    """
    try:
        from sqlalchemy import create_engine, text, inspect
        engine = create_engine(get_db_url(database), connect_args={"timeout": 10})
        inspector = inspect(engine)
        tables = inspector.get_table_names()

        if not tables:
            return "No tables found in database."

        schema_parts = []
        for table in tables:
            cols = inspector.get_columns(table)
            col_defs = ", ".join(
                f"{c['name']} {str(c['type'])}" for c in cols
            )
            schema_parts.append(f"CREATE TABLE {table} ({col_defs});")

            # Add sample rows for context
            try:
                with engine.connect() as conn:
                    sample = conn.execute(text(f"SELECT * FROM {table} LIMIT 3")).fetchall()  # noqa: S608
                    if sample:
                        schema_parts.append(f"-- Sample rows from {table}:")
                        for row in sample:
                            schema_parts.append(f"-- {dict(zip([c['name'] for c in cols], row))}")
            except Exception:
                pass

        return "\n".join(schema_parts)
    except ImportError:
        return "SQLAlchemy not installed. pip install sqlalchemy"
    except Exception as e:
        logger.warning("Schema extraction failed: %s", e)
        return f"Could not extract schema: {e}"


def generate_sql(
    question: str,
    schema: str,
    llm_fn: Any,
) -> str:
    """
    Use the LLM to generate a SQL SELECT query from a natural language question.

    Includes the schema so the LLM knows which tables and columns exist.
    Returns only the SQL (no explanation).
    """
    prompt = (
        "You are a SQL expert. Generate a single, correct SQL SELECT query for the question below.\n"
        "Rules:\n"
        "- Use ONLY SELECT (no INSERT, UPDATE, DELETE, DROP, CREATE, ALTER)\n"
        "- Return ONLY the SQL query, no explanation, no markdown fences\n"
        "- LIMIT results to 50 rows maximum\n"
        "- Use SQLite syntax\n\n"
        f"Database schema:\n{schema}\n\n"
        f"Question: {question}\n\n"
        "SQL query:"
    )
    try:
        raw = llm_fn(prompt).strip()
        # Strip markdown if present
        raw = re.sub(r"^```sql\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"^```\s*", "", raw)
        raw = re.sub(r"```\s*$", "", raw).strip()
        return raw
    except Exception as e:
        raise RuntimeError(f"SQL generation failed: {e}") from e


def execute_sql(sql: str, database: str | None = None, timeout: int = 10) -> list[dict]:
    """
    Execute a SQL query with safety checks.

    Only SELECT statements are allowed. Results are returned as a list of dicts.
    Raises ValueError for disallowed statements.
    """
    try:
        from sqlalchemy import create_engine, text
    except ImportError:
        raise ImportError("sqlalchemy not installed. pip install sqlalchemy") from None

    # Safety: only allow SELECT
    sql_clean = sql.strip().upper()
    if not sql_clean.startswith("SELECT"):
        raise ValueError(f"Only SELECT queries are allowed. Got: {sql[:50]}")

    # Block dangerous keywords
    dangerous = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE", "EXEC", "EXECUTE"]
    for keyword in dangerous:
        if re.search(rf"\b{keyword}\b", sql_clean):
            raise ValueError(f"Disallowed SQL keyword '{keyword}' detected.")

    try:
        engine = create_engine(get_db_url(database), connect_args={"timeout": timeout})
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            columns = list(result.keys())
            rows = [dict(zip(columns, row)) for row in result.fetchall()]
            return rows
    except Exception as e:
        raise RuntimeError(f"SQL execution failed: {e}") from e


def format_sql_results(rows: list[dict], question: str, sql: str) -> str:
    """
    Format SQL results as a RAG-friendly context string.

    Includes the question, the generated SQL, and the results in a table format.
    """
    if not rows:
        return f"SQL query returned no results.\nQuery: {sql}"

    # Build markdown table
    cols = list(rows[0].keys())
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join("---" for _ in cols) + " |"
    body_rows = []
    for row in rows[:50]:  # cap at 50 rows
        body_rows.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")

    table = "\n".join([header, separator] + body_rows)

    return (
        f"[SQL Result for: {question}]\n"
        f"Query executed: {sql}\n"
        f"Rows returned: {len(rows)}\n\n"
        f"{table}"
    )


def query_natural_language(
    question: str,
    database: str | None = None,
    llm_fn: Any = None,
) -> str:
    """
    Full text-to-SQL pipeline: question → SQL → execute → formatted result.

    Args:
        question: natural language question
        database: database URL or path (uses SQL_DATABASE_URL from config if not provided)
        llm_fn: LLM callable for SQL generation (auto-uses configured backend if None)

    Returns:
        Formatted string of SQL results, suitable for RAG context
    """
    if llm_fn is None:
        try:
            from core.generation import get_backend
            llm_fn = get_backend().complete_raw
        except Exception as e:
            return f"SQL retrieval requires an LLM backend: {e}"

    schema = get_schema(database)
    if schema.startswith("No tables") or schema.startswith("Could not") or schema.startswith("SQLAlchemy"):
        return schema

    try:
        sql = generate_sql(question, schema, llm_fn)
        logger.info("Generated SQL: %s", sql[:200])

        rows = execute_sql(sql, database)
        result = format_sql_results(rows, question, sql)
        logger.info("SQL query returned %d rows", len(rows))
        return result

    except (ValueError, RuntimeError) as e:
        logger.warning("SQL retrieval failed for '%s': %s", question, e)
        return f"SQL error: {e}"


# ── Sample database creation (for demos) ─────────────────────────────────────


def create_sample_db() -> Path:
    """
    Create a sample SQLite database with realistic business data.

    Creates 3 tables: products, customers, orders.
    Useful for demoing text-to-SQL without a real database.
    """
    try:
        from sqlalchemy import create_engine, text
    except ImportError:
        raise ImportError("pip install sqlalchemy") from None

    DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{DEFAULT_DB_PATH}")

    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS orders"))
        conn.execute(text("DROP TABLE IF EXISTS products"))
        conn.execute(text("DROP TABLE IF EXISTS customers"))

        conn.execute(text("""
            CREATE TABLE products (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT,
                price REAL,
                revenue_q1 REAL,
                revenue_q2 REAL,
                revenue_q3 REAL,
                revenue_q4 REAL,
                in_stock INTEGER DEFAULT 1
            )
        """))

        conn.execute(text("""
            CREATE TABLE customers (
                id INTEGER PRIMARY KEY,
                name TEXT,
                region TEXT,
                tier TEXT,
                total_spend REAL,
                joined_date TEXT
            )
        """))

        conn.execute(text("""
            CREATE TABLE orders (
                id INTEGER PRIMARY KEY,
                customer_id INTEGER,
                product_id INTEGER,
                quantity INTEGER,
                total REAL,
                order_date TEXT,
                status TEXT,
                FOREIGN KEY(customer_id) REFERENCES customers(id),
                FOREIGN KEY(product_id) REFERENCES products(id)
            )
        """))

        # Sample data
        products = [
            (1, "Enterprise Plan", "SaaS", 999.0, 234000, 289000, 312000, 401000, 1),
            (2, "Pro Plan", "SaaS", 99.0, 45000, 52000, 61000, 78000, 1),
            (3, "Starter Plan", "SaaS", 9.0, 12000, 14000, 15000, 18000, 1),
            (4, "Data Connector", "Add-on", 199.0, 23000, 31000, 28000, 42000, 1),
            (5, "API Access", "Add-on", 299.0, 18000, 22000, 35000, 44000, 1),
        ]
        conn.execute(text("INSERT INTO products VALUES (?,?,?,?,?,?,?,?,?)"), products)

        customers = [
            (1, "Acme Corp", "North America", "enterprise", 1200000, "2022-01-15"),
            (2, "TechStart Inc", "Europe", "pro", 45000, "2023-03-22"),
            (3, "GlobalData Ltd", "APAC", "enterprise", 890000, "2021-07-08"),
            (4, "Innovate LLC", "North America", "starter", 2400, "2024-01-01"),
            (5, "DataViz Co", "Europe", "pro", 67000, "2023-06-14"),
        ]
        conn.execute(text("INSERT INTO customers VALUES (?,?,?,?,?,?)"), customers)

        orders = [
            (1, 1, 1, 12, 11988.0, "2024-01-15", "completed"),
            (2, 2, 2, 5, 495.0, "2024-02-01", "completed"),
            (3, 3, 1, 8, 7992.0, "2024-02-15", "completed"),
            (4, 4, 3, 1, 9.0, "2024-03-01", "completed"),
            (5, 5, 2, 3, 297.0, "2024-03-15", "completed"),
            (6, 1, 4, 2, 398.0, "2024-04-01", "completed"),
            (7, 3, 5, 4, 1196.0, "2024-04-15", "pending"),
        ]
        conn.execute(text("INSERT INTO orders VALUES (?,?,?,?,?,?,?)"), orders)
        conn.commit()

    logger.info("Sample database created at '%s'", DEFAULT_DB_PATH)
    return DEFAULT_DB_PATH
