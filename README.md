# An Integrated Approach to Blockchain Asset Modeling: LSTM for Feature Extraction and Deep Q-Learning for Reinforcement Prediction

## Disclaimer

This project, "An Integrated Approach to Blockchain Asset Modeling," is provided for educational and research purposes only. It is not intended for use in actual trading or financial decision-making. The authors and contributors offer this project as a demonstration of data analysis and machine learning techniques applied to blockchain asset data.

The content and code within this project are not financial advice and should not be interpreted as such. Users are solely responsible for any actions taken based on the information provided in this project. The authors and contributors make no representations or warranties about the accuracy, reliability, completeness, or timeliness of the content or code.

Any application of the techniques, ideas, or code in this project is at the user's own risk. The authors and contributors expressly disclaim all liability for any financial losses, damages, or other negative consequences that may result from such use.

Users are advised to conduct their own research and consult with professional financial advisors before making any investment decisions. The laws and regulations regarding financial trading and investments vary across jurisdictions, and users are responsible for ensuring their compliance with such laws in their respective localities.

This project does not promote, encourage, or endorse any form of financial speculation or gambling. The authors and contributors are not liable for any misuse of the project’s content or code.

By using this project, you acknowledge and agree to these terms.

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
