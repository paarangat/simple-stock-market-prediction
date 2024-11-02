
# Simple Stock Market Prediction using Machine Learning

## Overview
This project aims to predict the stock market performance of the NIFTY 50 index using historical data from Yahoo Finance. We utilize multiple machine learning models for both regression and classification to predict stock prices and their movements. The project demonstrates the use of several models and their comparison to identify the best-performing approach.

## Features
- **Regression Models**: Linear Regression, Support Vector Regression (SVR), K-Nearest Neighbors (KNN), Polynomial Regression, and Random Forest.
- **Classification Models**: Logistic Regression, SVM, KNN, Naive Bayes, and Random Forest for predicting the direction of stock price movement.
- **Feature Engineering**: Added features from the 'Date' column, such as Year, Month, and Day, and included Volume to improve model accuracy.
- **Visualizations**: Scatter plots, regression lines, residual plots, and confusion matrices to evaluate and compare model performance.

## Dataset
The dataset used is historical data of the NIFTY 50 index for the past 5 years, collected from Yahoo Finance. The data includes the following features:
- **Date**: Date of the record
- **Open, High, Low, Close**: Price indicators for each trading day
- **Volume**: Volume of shares traded

## Requirements
To run the project, you need to install the following Python libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

You can install these dependencies using the following command:
```sh
pip install numpy pandas matplotlib seaborn scikit-learn
```

## How to Run
1. **Clone the Repository**:
   ```sh
   git clone https://github.com/paarangat/stock-market-prediction.git
   ```
2. **Navigate to the Project Directory**:
   ```sh
   cd stock-market-prediction
   ```
3. **Run the Jupyter Notebook**:
   - Open the `Stock_Market_Prediction_Project.ipynb` notebook in Jupyter Notebook or any compatible IDE.
   - Execute the cells sequentially to see the pre-processing steps, model training, and visualizations.

## Project Structure
- **`nifty50_data_5years.csv`**: Historical stock data used for prediction.
- **`Stock_Market_Prediction_Project.ipynb`**: Main notebook containing all steps from data pre-processing to model evaluation.
- **`related_work_stock_prediction.docx`**: Document describing similar projects and how this project is different.

## Model Evaluation
- **Regression Models**: Evaluated using metrics such as Mean Squared Error (MSE) and R² score. We compared predicted vs. actual prices using scatter plots and regression lines.
- **Classification Models**: Evaluated using accuracy, confusion matrices, precision, recall, and F1 score to understand the model's performance.

## Key Findings
- **Support Vector Regression (SVR)** and **Random Forest** showed better performance for non-linear relationships in stock price prediction.
- **Random Forest Classifier** performed well in predicting stock movement direction compared to other classifiers.

## Future Improvements
- **Hyperparameter Tuning**: Use GridSearchCV to optimize model parameters for better results.
- **Cross-Validation**: Implement cross-validation to improve model robustness.
- **Additional Features**: Consider adding more financial indicators like Moving Averages (MA) and Relative Strength Index (RSI) to enhance prediction accuracy.

## Contributing
Feel free to submit pull requests or open issues for improvements or new features. Contributions are always welcome!

## Contact
For questions or suggestions, please contact Paarangat Rai Sharma at paarangatprs@gmail.com.

