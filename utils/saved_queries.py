# variables
CATALOG = "workspace"
QUERIES_TABLE = f"{CATALOG}.careerpulse_gold.saved_queries"

# imports
from pyspark.sql.functions import col, current_timestamp


def upsert_saved_query(
    spark,
    query_id: str, 
    query_title: str, 
    query_pattern: str, 
    is_active: bool = True
    ) -> None:
    """
    Idempotent upsert into saved_queries keyed by query_id.
    If query_id exists, update fields; else insert.
    """
    incoming = spark.createDataFrame([{
        "query_id": query_id,
        "query_title": query_title,
        "query_pattern": query_pattern,
        "is_active": is_active
    }])

    incoming.createOrReplaceTempView("incoming_query")

    spark.sql(f"""
    MERGE INTO {QUERIES_TABLE} AS t
    USING incoming_query AS s
    ON t.query_id = s.query_id
    WHEN MATCHED THEN UPDATE SET
      t.query_title   = s.query_title,
      t.query_pattern = s.query_pattern,
      t.is_active     = s.is_active
    WHEN NOT MATCHED THEN INSERT (
      query_id, query_title, query_pattern, is_active, created_at
    ) VALUES (
      s.query_id, s.query_title, s.query_pattern, s.is_active, current_timestamp()
    )
    """)
