# E-commerce Customer Analysis

A machine learning project that analyzes customer behavior and predicts yearly spending for an e-commerce platform using linear regression.

## Overview

This project analyzes customer data to understand the relationship between various user engagement metrics and their yearly spending. The analysis includes data exploration, visualization, and predictive modeling to help businesses optimize their customer experience and revenue.

## Dataset

The analysis uses an e-commerce dataset (`ecommerce.csv`) containing the following features:

- **Avg. Session Length**: Average time spent per session on the platform
- **Time on App**: Time spent on the mobile application
- **Time on Website**: Time spent on the website
- **Length of Membership**: Duration of customer membership
- **Yearly Amount Spent**: Target variable - total yearly spending (to be predicted)

## Requirements

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
pylab
```

## Installation

1. Clone this repository
2. Install required packages:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn scipy
   ```
3. Ensure you have the `ecommerce.csv` file in the same directory as the script

## Analysis Workflow

### 1. Data Exploration
- Load and examine the dataset structure
- Check for missing values and duplicates
- Generate descriptive statistics

### 2. Data Visualization
- **Pairplot**: Comprehensive visualization of relationships between all features
- **Scatter Plot**: Specific analysis of Time on App vs Yearly Amount Spent
- **Residual Analysis**: Distribution and normality testing of model residuals

### 3. Machine Learning Model
- **Model**: Linear Regression
- **Features**: Avg. Session Length, Time on App, Time on Website, Length of Membership
- **Target**: Yearly Amount Spent
- **Train-Test Split**: 80% training, 20% testing

### 4. Model Evaluation
The model performance is evaluated using:
- **Mean Absolute Error (MAE)**: Average absolute difference between predicted and actual values
- **Root Mean Squared Error (RMSE)**: Square root of average squared differences
- **R² Score**: Coefficient of determination (explained variance)

## Key Visualizations

1. **Pairplot**: Shows correlations between all numerical features
2. **Predicted vs Actual**: Scatter plot comparing model predictions with actual values
3. **Residuals Distribution**: Histogram showing residual distribution
4. **Q-Q Plot**: Tests normality assumption of residuals

## Usage

```python
# Run the complete analysis
python ecomence.py
```

The script will:
1. Load and explore the data
2. Generate visualizations
3. Train the linear regression model
4. Display evaluation metrics
5. Show residual analysis plots

## Results Interpretation

- **Higher R² Score**: Indicates better model performance (closer to 1.0)
- **Lower MAE/RMSE**: Indicates more accurate predictions
- **Normal Residuals**: Validates linear regression assumptions

## Business Insights

The analysis helps identify:
- Which customer engagement metrics most strongly predict spending
- Optimal areas for platform improvement (app vs website)
- Customer segmentation opportunities based on usage patterns

## Future Enhancements

- Feature engineering for better predictive power
- Advanced regression techniques (Ridge, Lasso)
- Customer segmentation analysis
- Time series analysis for seasonal patterns
- A/B testing framework integration

## File Structure

```
├── ecomence.py          # Main analysis script
├── ecommerce.csv        # Dataset (not included)
├── README.md           # This file
└── requirements.txt    # Python dependencies
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is open source and available under the MIT License.
