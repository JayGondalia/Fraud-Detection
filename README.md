# Fraud Detection

A machine learning project for detecting fraudulent credit card transactions using various classification algorithms.

## Overview

This project implements fraud detection models to identify fraudulent credit card transactions. The system analyzes transaction patterns and characteristics to flag suspicious activities, helping to minimize financial losses and protect customers.

## Repository Structure
```
Fraud-Detection/
├── Credit_Card.csv                                    # Credit card transaction dataset
├── Credit_Card_Fraud_Dataset.csv                      # Additional fraud dataset
├── Transact_Guard_Fraud_Risk_Modeling_Using_ml.ipynb  # Main analysis notebook
├── LICENSE                                            # MIT License
└── README.md                                          # Project documentation
```

## Dataset

The project uses credit card transaction datasets containing various features such as:
- Transaction amount
- Transaction time
- Anonymized features (V1, V2, ..., V28)
- Class label (0 = legitimate, 1 = fraudulent)

The datasets are highly imbalanced, with fraudulent transactions representing a small percentage of total transactions.

## Features

- **Data Exploration**: Comprehensive analysis of transaction patterns and fraud characteristics
- **Data Preprocessing**: Handling missing values, feature scaling, and dealing with class imbalance
- **Feature Engineering**: Creating relevant features to improve model performance
- **Model Training**: Implementation of multiple machine learning algorithms
- **Model Evaluation**: Performance assessment using appropriate metrics for imbalanced datasets
- **Visualization**: Clear visualizations of data distributions and model results

## Machine Learning Models

The project explores various classification algorithms including:
- Logistic Regression
- Decision Trees
- Random Forest
- Gradient Boosting
- XGBoost
- Neural Networks

## Evaluation Metrics

Given the imbalanced nature of fraud detection, the following metrics are used:
- **Precision**: Accuracy of positive predictions
- **Recall**: Coverage of actual positive cases
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Overall model performance
- **Confusion Matrix**: Detailed prediction breakdown

## Getting Started

### Prerequisites
```bash
Python 3.7+
Jupyter Notebook
```

### Required Libraries
```bash
pandas
numpy
scikit-learn
matplotlib
seaborn
imbalanced-learn
xgboost
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/JayGondalia/Fraud-Detection.git
cd Fraud-Detection
```

2. Install required packages:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn xgboost jupyter
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

4. Open `Transact_Guard_Fraud_Risk_Modeling_Using_ml.ipynb` and run the cells

## Usage

1. **Data Loading**: Load the credit card transaction datasets
2. **Exploratory Data Analysis**: Analyze data distributions and patterns
3. **Preprocessing**: Clean and prepare data for modeling
4. **Model Training**: Train various classification models
5. **Evaluation**: Compare model performance using multiple metrics
6. **Prediction**: Use the best model to detect fraudulent transactions

## Results

The notebook includes detailed analysis of:
- Data distribution and patterns
- Model performance comparisons
- Feature importance analysis
- Fraud detection accuracy metrics

## Handling Imbalanced Data

The project employs several techniques to handle class imbalance:
- SMOTE (Synthetic Minority Over-sampling Technique)
- Random Under-sampling
- Class weight adjustment
- Ensemble methods

## Future Improvements

- Real-time fraud detection implementation
- Deep learning approaches (LSTM, Autoencoders)
- Additional feature engineering
- Model deployment as API
- Integration with transaction monitoring systems

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Jay Gondalia

## Acknowledgments

- Dataset sources for credit card fraud detection
- Scikit-learn and imbalanced-learn libraries
- Machine learning community for fraud detection research

---

**Note**: This project is for educational purposes. In production environments, fraud detection systems require additional security measures, real-time processing capabilities, and regulatory compliance.
