"""Conceptual point-in-time join. Simulates what a feature store does with pandas."""

import pandas as pd


def main() -> None:
    events = pd.DataFrame({
        "user_id": [1, 1, 2, 2, 3],
        "event_time": pd.to_datetime([
            "2026-01-01", "2026-02-01", "2026-01-15", "2026-02-15", "2026-01-20",
        ]),
        "label": [1, 0, 1, 0, 1],
    })

    features = pd.DataFrame({
        "user_id": [1, 1, 2, 2, 3],
        "feature_time": pd.to_datetime([
            "2025-12-15", "2026-01-20", "2026-01-05", "2026-02-10", "2026-01-18",
        ]),
        "purchases_30d": [5, 8, 3, 7, 2],
    })

    # point-in-time join: for each event, find the latest feature_time <= event_time
    events = events.sort_values("event_time")
    features = features.sort_values("feature_time")
    joined = pd.merge_asof(
        events, features,
        left_on="event_time", right_on="feature_time",
        by="user_id",
        direction="backward",
    )
    print(joined)


if __name__ == "__main__":
    main()
