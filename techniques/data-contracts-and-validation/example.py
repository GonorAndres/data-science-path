"""Pandera schema-based validation example."""

import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series


class OrderSchema(pa.DataFrameModel):
    order_id: Series[str] = pa.Field(unique=True)
    customer_id: Series[str]
    amount: Series[float] = pa.Field(ge=0)
    status: Series[str] = pa.Field(isin=["created", "paid", "cancelled", "delivered"])


@pa.check_types
def process_orders(df: DataFrame[OrderSchema]) -> DataFrame[OrderSchema]:
    return df[df["amount"] > 0]


def main() -> None:
    good = pd.DataFrame({
        "order_id": ["o1", "o2", "o3"],
        "customer_id": ["c1", "c2", "c1"],
        "amount": [10.0, 25.0, 0.5],
        "status": ["paid", "delivered", "created"],
    })
    print(process_orders(good))

    try:
        bad = pd.DataFrame({
            "order_id": ["o1", "o1"],
            "customer_id": ["c1", "c1"],
            "amount": [10.0, -1.0],
            "status": ["paid", "purgatory"],
        })
        process_orders(bad)
    except pa.errors.SchemaError as e:
        print("schema violation detected:", e)


if __name__ == "__main__":
    main()
