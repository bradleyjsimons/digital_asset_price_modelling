from src.data import data_controller


# Define date parameters
start_date = "2020-01-01"
end_date = "2023-01-01"

# Fetch and prep the data
df = data_controller.main(start_date, end_date)
