# BTC Price Prediction: Integrating Blockchain and Market Data with Reinforcement Learning

## Description

This project aims to predict Bitcoin prices using a combination of blockchain metrics, market data, and reinforcement learning algorithms. The current progress includes:

- Retrieval of Bitcoin candle data from Yahoo Finance.
- Retrieval of blockchain metrics such as mempool size, total hash rate, miner revenue, and others from blockchain.com's charts API.
- Addition of various technical indicators like RSI, CCI, ATR, on-balance volume, stochastic oscillators, etc., into a comprehensive Pandas DataFrame.
- Data preprocessing and cleaning.
- Feature extraction using an LSTM neural network, with the extracted features added to the dataset.

The next steps will focus on feature selection to determine the most impactful features for our modeling.

## Installation

To set up this project:

1. Clone the repository:

   ```
   git clone [repository URL]
   ```

2. Navigate to the project directory:

   ```
   cd [project directory]
   ```

3. Create a Conda environment:

   ```
   conda create --name myenv python=3.x
   ```

4. Activate the environment:

   ```
   conda activate myenv
   ```

5. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the project using:

```
python main.py
```

(Detailed usage instructions will be updated as the project evolves.)

## Contributing

Contributions are welcome, particularly in areas like model improvement, data processing, and code optimization. Please submit a pull request with your proposed changes.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Contact

bradleyjsimons@velveteen.ai
