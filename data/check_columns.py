import glob
import os

import pandas as pd
import numpy as np


def null_report(df):
    """Report null/NA values per column of a single table.

    For every column returns:
      - column name
      - number of null or NA values
      - percentage of those null values relative to the table's row count
      - number of duplicate values in the column

    Parameters
    ----------
    df : pd.DataFrame
        A single table.

    Returns
    -------
    pd.DataFrame
        One row per column with the columns:
        ['column', 'null_count', 'null_percentage', 'duplicate_count'].
    """
    total_rows = len(df)

    report = pd.DataFrame({
        "column": df.columns,
        "null_count": df.isna().sum().values,
        "duplicate_count": [int(df[col].duplicated().sum()) for col in df.columns],
    })

    report["null_percentage"] = np.where(
        total_rows > 0,
        report["null_count"] / total_rows * 100,
        0.0,
    ).round(2)

    return report.reset_index(drop=True)


# Expected values per table, derived from table_info.md (row counts ignored).
# Each entry maps a column name to a predicate the value must satisfy, or None
# when no specific rule is defined. Used by the completeness/consistency check.
TABLE_EXPECTATIONS = {
    "olist_orders_dataset": {
        "order_id": "not_null",
        "customer_id": "not_null",
        "order_status": lambda s: s.isin(
            ["created", "approved", "invoiced", "processing",
             "shipped", "delivered", "unavailable", "canceled"]
        ),
        "order_purchase_timestamp": "not_null",
        "order_approved_at": None,
        "order_delivered_carrier_date": None,
        "order_delivered_customer_date": None,
        "order_estimated_delivery_date": "not_null",
    },
    "olist_order_items_dataset": {
        "order_id": "not_null",
        "order_item_id": "not_null",
        "product_id": "not_null",
        "seller_id": "not_null",
        "shipping_limit_date": "not_null",
        "price": lambda s: pd.to_numeric(s, errors="coerce").ge(0),
        "freight_value": lambda s: pd.to_numeric(s, errors="coerce").ge(0),
    },
    "olist_order_payments_dataset": {
        "order_id": "not_null",
        "payment_sequential": "not_null",
        "payment_type": lambda s: s.isin(
            ["credit_card", "boleto", "voucher", "debit_card", "not_defined"]
        ),
        "payment_installments": lambda s: pd.to_numeric(s, errors="coerce").ge(0),
        "payment_value": lambda s: pd.to_numeric(s, errors="coerce").ge(0),
    },
    "olist_order_reviews_dataset": {
        "review_id": "not_null",
        "order_id": "not_null",
        "review_score": lambda s: pd.to_numeric(s, errors="coerce").between(1, 5),
        "review_comment_title": None,
        "review_comment_message": None,
        "review_creation_date": "not_null",
        "review_answer_timestamp": "not_null",
    },
    "olist_customers_dataset": {
        "customer_id": "not_null",
        "customer_unique_id": "not_null",
        "customer_zip_code_prefix": "not_null",
        "customer_city": "not_null",
        "customer_state": "not_null",
    },
    "olist_products_dataset": {
        "product_id": "not_null",
        "product_category_name": None,
        "product_name_lenght": None,
        "product_description_lenght": None,
        "product_photos_qty": None,
        "product_weight_g": None,
        "product_length_cm": None,
        "product_height_cm": None,
        "product_width_cm": None,
    },
    "olist_sellers_dataset": {
        "seller_id": "not_null",
        "seller_zip_code_prefix": "not_null",
        "seller_city": "not_null",
        "seller_state": "not_null",
    },
    "olist_geolocation_dataset": {
        "geolocation_zip_code_prefix": "not_null",
        "geolocation_lat": "not_null",
        "geolocation_lng": "not_null",
        "geolocation_city": "not_null",
        "geolocation_state": "not_null",
    },
    "product_category_name_translation": {
        "product_category_name": "not_null",
        "product_category_name_english": "not_null",
    },
}


def _satisfies(series, rule):
    """Return a boolean Series telling whether each value meets ``rule``."""
    if rule is None:
        return pd.Series(True, index=series.index)
    if rule == "not_null":
        return series.notna()
    return rule(series).fillna(False)


def check_table(df, table_name, expectations=None):
    """Report the completeness and consistency of the columns of one table.

    Parameters
    ----------
    df : pd.DataFrame
        A single table.
    table_name : str
        Name of the table (key used to look up expectations).
    expectations : dict, optional
        Mapping of column name -> rule. Defaults to ``TABLE_EXPECTATIONS``.
        A rule is either ``"not_null"``, a callable returning a boolean Series,
        or ``None`` (no specific rule).

    Returns
    -------
    pd.DataFrame
        One row per column with:
        ['column', 'total', 'non_null', 'completeness_pct',
         'rule_satisfied', 'violations', 'consistency_pct', 'consistent'].
    """
    if expectations is None:
        expectations = TABLE_EXPECTATIONS
    rules = expectations.get(table_name, {})

    total = len(df)
    rows = []
    for col in df.columns:
        series = df[col]
        non_null = int(series.notna().sum())
        completeness = round(non_null / total * 100, 2) if total else 0.0

        rule = rules.get(col, None)
        if rule is None:
            satisfied = pd.Series(True, index=series.index)
        else:
            satisfied = _satisfies(series, rule) & series.notna()

        violations = int((~satisfied).sum())
        consistency = round(satisfied.sum() / total * 100, 2) if total else 0.0

        rows.append({
            "column": col,
            "total": total,
            "non_null": non_null,
            "completeness_pct": completeness,
            "rule_satisfied": int(satisfied.sum()),
            "violations": violations,
            "consistency_pct": consistency,
            "consistent": violations == 0,
        })

    return pd.DataFrame(rows)


def check_all_tables(folder=None, pattern="*.csv", expectations=None):
    """Run ``check_table`` on every .csv in a folder.

    Parameters
    ----------
    folder : str, optional
        Folder to scan. Defaults to the folder containing this script.
    pattern : str, optional
        Glob pattern for the files to load (default ``"*.csv"``).
    expectations : dict, optional
        Passed through to ``check_table``.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of table name (file name without extension) -> report.
    """
    if folder is None:
        folder = os.path.dirname(os.path.abspath(__file__))

    reports = {}
    for path in sorted(glob.glob(os.path.join(folder, pattern))):
        table_name = os.path.splitext(os.path.basename(path))[0]
        reports[table_name] = check_table(
            pd.read_csv(path), table_name, expectations=expectations
        )
    return reports


def table_datatypes(tables):
    """Return a dataframe with the datatypes of every column of every table.

    Parameters
    ----------
    tables : dict[str, pd.DataFrame]
        Mapping of table name -> dataframe.

    Returns
    -------
    pd.DataFrame
        One row per (table, column) with ['table', 'column', 'dtype'].
    """
    rows = []
    for name, df in tables.items():
        for col in df.columns:
            rows.append({
                "table": name,
                "column": col,
                "dtype": str(df[col].dtype),
            })
    return pd.DataFrame(rows)
