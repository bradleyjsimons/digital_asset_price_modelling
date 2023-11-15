from src.data import (
    fetch_bitcoin_data,
    clean_data,
    add_all_technical_indicators,
    add_blockchain_data,
)


# Define date parameters
start_date = "2020-01-01"
end_date = "2023-01-01"

# Fetch and process Bitcoin data
df = fetch_bitcoin_data(start_date, end_date)
df = add_blockchain_data(df, timespan="3years", start=start_date)

# add indicators
df = add_all_technical_indicators(df)

# clean data
df = clean_data(df)

print(df)
