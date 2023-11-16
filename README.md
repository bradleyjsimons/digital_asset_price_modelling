# An Integrated Approach to Blockchain Asset Modeling: LSTM for Feature Extraction and Deep Q-Learning for Reinforcement Prediction

## Disclaimer

This project, "An Integrated Approach to Blockchain Asset Modeling," is provided for educational and research purposes only. It is not intended for use in actual trading or financial decision-making. The authors and contributors offer this project as a demonstration of data analysis and machine learning techniques applied to blockchain asset data.

The content and code within this project are not financial advice and should not be interpreted as such. Users are solely responsible for any actions taken based on the information provided in this project. The authors and contributors make no representations or warranties about the accuracy, reliability, completeness, or timeliness of the content or code.

Any application of the techniques, ideas, or code in this project is at the user's own risk. The authors and contributors expressly disclaim all liability for any financial losses, damages, or other negative consequences that may result from such use.

Users are advised to conduct their own research and consult with professional financial advisors before making any investment decisions. The laws and regulations regarding financial trading and investments vary across jurisdictions, and users are responsible for ensuring their compliance with such laws in their respective localities.

This project does not promote, encourage, or endorse any form of financial speculation or gambling. The authors and contributors are not liable for any misuse of the project’s content or code.

By using this project, you acknowledge and agree to these terms.

## Description

This project represents a comprehensive approach to predicting Bitcoin prices, integrating blockchain metrics, market data, and advanced machine learning techniques. The current progress and methodologies include:

- **Data Retrieval**: Gathering Bitcoin OHLCV (Open, High, Low, Close, Volume) data from Yahoo Finance.
- **Log Returns Calculation**: Calculating and including log returns as part of the feature set immediately after data retrieval from Yahoo Finance.
- **Blockchain Metrics**: Incorporating blockchain-specific data such as hash rate, average block size, network difficulty, miner revenue, and mempool size, sourced from blockchain.com's charts API.
- **Technical Analysis Indicators**: Enhancing the dataset with a variety of technical indicators, including Bollinger Bands, Stochastic Oscillators, MACD, RSI, Simple Moving Average (SMA), Exponential Moving Average (EMA), Average True Range (ATR), MACD Histogram, On-Balance Volume, and Commodity Channel Index (CCI).
- **Data Preprocessing**: Performing essential data preprocessing, which includes normalization (using MinMaxScaler), forward-filling missing values, and removing duplicates. The scalar used for normalization is stored for potential inverse transformation in later stages.
- **Feature Extraction with LSTM**: Utilizing an LSTM neural network to extract additional features, adding a new dimension to the dataset.
- **Target Variable**: Introducing a binary target variable that indicates whether the closing price of Bitcoin goes up or down, facilitating a predictive modeling approach.
- **Data Cleaning**: Systematic data cleaning throughout the process ensures data integrity and usability.
- **Dataset Storage**: Post-processing, the complete dataset is stored as a CSV file for easy access and reproducibility.
- **Deep Q-Network Implementation**: Currently, the focus is on implementing a Deep Q-Network (DQN) reinforcement learning algorithm to develop a sophisticated trading strategy based on the processed data. This involves setting up an appropriate trading environment and fine-tuning the model.
- **Future Steps**: The next phase will involve rigorous backtesting of the model to evaluate its effectiveness and refine its predictive capabilities.

This project encapsulates a blend of data science and machine learning techniques aimed at modeling and understanding the complex dynamics of Bitcoin prices, offering insights into the potential of combining traditional financial analysis with cutting-edge AI methodologies.

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
