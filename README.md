# Utilizing Deep Q Network Agents for Reinforcement Learning in Blockchain Asset Forecasting

## Current Status/Latest Update

- Completed data collection, preprocessing, and feature extraction.
- Developed and tested a Deep Q Network model for forecasting.
- Performed backtesting and visualizations, analyzing metrics like Sharpe ratio and maximum drawdown.
- Next steps include building infrastructure for extensive hyperparameter tuning and conducting numerous tests to compare and optimize models.

# Financial Disclaimer for "An Integrated Approach to Blockchain Asset Modeling"

**IMPORTANT NOTICE**: The project "An Integrated Approach to Blockchain Asset Modeling" is provided exclusively for educational and research purposes. It is not intended for use in actual trading or financial decision-making. The content and code within this project are for demonstration of data analysis and machine learning techniques applied to blockchain asset data and should not be construed as financial advice.

By using this software, you agree to hold harmless and exempt the authors and contributors from any claims, damages, losses, or legal responsibilities arising from its use. The authors and contributors make no representations or warranties regarding the accuracy, reliability, completeness, or timeliness of the content or code and expressly disclaim all liability for any financial losses, damages, or other negative consequences that may result from such use.

Any application of the techniques, ideas, or code in this project is at the user's own risk. Users are solely responsible for any actions taken based on the information provided in this project and are advised to conduct their own research and consult with professional financial advisors before making any investment decisions.

The laws and regulations regarding financial trading and investments vary across jurisdictions, and users are responsible for ensuring their compliance with such laws in their respective localities. This project does not promote, encourage, or endorse any form of financial speculation or gambling, and the authors and contributors are not liable for any misuse of the project’s content or code.

By using this project, you acknowledge and agree to these terms.

## Description

This project represents a sophisticated approach to predicting Bitcoin prices, integrating blockchain metrics, market data, and advanced machine learning techniques. Key developments and methodologies include:

- **Data Retrieval and Log Returns Calculation**: Gathering Bitcoin OHLCV (Open, High, Low, Close, Volume) data from Yahoo Finance and calculating log returns.
- **Blockchain Metrics**: Adding blockchain-specific data such as hash rate, average block size, network difficulty, miner revenue, and mempool size, sourced from blockchain.com's charts API.
- **Technical Analysis Indicators**: Incorporating a variety of technical indicators like Bollinger Bands, Stochastic Oscillators, MACD, RSI, SMA, EMA, ATR, MACD Histogram, On-Balance Volume, and CCI.
- **Data Preprocessing**: Implementing essential preprocessing steps, including normalization using MinMaxScaler, forward-filling missing values, and removing duplicates.
- **Feature Extraction with LSTM**: Using an LSTM neural network to derive additional features, contributing a new dimension to the dataset.
- **Target Variable Addition**: Introducing a binary target variable to indicate whether the closing price of Bitcoin goes up or down.
- **Dataset Storage**: Storing the processed dataset as a CSV file for accessibility and reproducibility.
- **Reinforcement Learning Environment Enhancement**:
  - **Fee Calculation**: Integrating a fee calculation based on the Kraken API fee structure into the step function of the reinforcement learning environment.
  - **Value Function and Environment Update**: Incorporating a balance metric and optimizing for the highest return relative to the initial balance.
- **Deep Q-Network (DQN) Model Update**:
  - **Model Architecture**: Updating the DQN model to include two layers of 64 neurons each, followed by a third layer of 32 neurons, all using ReLU activation, and a final dense layer with linear activation.
  - **Optimization and Loss**: Employing the Adam optimizer with mean squared error as the loss function.
- **Future Steps**:
  - **Model Testing and Backtesting**: Continuing to test the code, perform backtesting, and refine the model based on results.
  - **Performance Analysis and Visualization**: Planning to provide charts, visualizations, and analyze metrics like the Sharpe ratio and maximum drawdown to evaluate the model's effectiveness.
  - **Ongoing Development**: Further code testing and enhancements to improve predictive accuracy and reliability.

This project encapsulates a blend of data science, machine learning, and financial analysis techniques, aimed at understanding the complex dynamics of Bitcoin prices. It offers insights into the potential of combining traditional financial analysis with cutting-edge AI methodologies.

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
