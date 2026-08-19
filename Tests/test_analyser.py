import numpy as np
import pandas as pd
import pytest

from Stocks_Full_Stack import model


def make_stock_dataframe(rows: int) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-01", periods=rows)

    close = np.linspace(100.0, 100.0 + rows - 1, rows)

    return pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.full(rows, 1_000_000),
        },
        index=index,
    )


def test_test_period_is_strictly_after_training_period():
    """
    A next-day forecasting model must train on past dates and test on later dates.
    """
    raw = make_stock_dataframe(rows=300)
    indicators = model.add_indicators(raw)

    X_train, X_test, _, _, _ = model.prepare_training_data(
        indicators,
        test_size=0.30,
    )

    assert X_train.index.max() < X_test.index.min()


def test_200_days_cannot_produce_a_supervised_training_row():
    """
    Volatility200 needs 200 returns, and target needs one later Close price.
    """
    raw = make_stock_dataframe(rows=200)
    indicators = model.add_indicators(raw)

    with pytest.raises(ValueError, match="Not enough"):
        model.prepare_training_data(indicators)

def test_unsorted_dates_are_rejected():
    """
    pct_change(), rolling(), and shift(-1) only make chronological sense
    if the index is already ordered.
    """
    raw = make_stock_dataframe(rows=300)

    unsorted = raw.sample(frac=1.0, random_state=42)

    with pytest.raises(ValueError, match="sorted|chronological"):
        model.prepare_training_data(model.add_indicators(unsorted))

#def test_at_least_two_test_rows_are_required_for_r2():
 #   """
  #  R2 is undefined for a single test observation.
   # """
    #raw = make_stock_dataframe(rows=300)
    #indicators = model.add_indicators(raw)

    #X_train, X_test, y_train, y_test, _ = model.prepare_training_data(
     #   indicators,
        test_size=0.01,
    #)

    #assert len(y_test) >= 2