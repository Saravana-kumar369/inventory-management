# inventory-management
## description

## Program
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Global variable for the trained model
model = None

def load_data(file_path):
    """Load the sales data from a CSV file."""
    try:
        data = pd.read_csv(file_path, encoding='ISO-8859-1', parse_dates=['order date (DateOrders)', 'shipping date (DateOrders)'])
        print("Data loaded successfully.")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    """Perform data preprocessing."""
    columns_to_drop = [
        'Customer Password', 'Customer Email', 'Customer Fname', 'Customer Lname',
        'Product Description', 'Product Image', 'Customer Street', 'Customer Zipcode'
    ]
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    print("Dropped unnecessary columns.")

    # Handle missing values
    df.fillna({
        'Benefit per order': df['Benefit per order'].mean(),
        'Sales per customer': df['Sales per customer'].mean(),
        'Late_delivery_risk': df['Late_delivery_risk'].mode()[0],
        'Category Name': df['Category Name'].mode()[0],
        'Customer City': df['Customer City'].mode()[0],
        'Customer Country': df['Customer Country'].mode()[0],
        'Customer Segment': df['Customer Segment'].mode()[0],
        'Department Name': df['Department Name'].mode()[0],
        'Order Region': df['Order Region'].mode()[0],
        'Order State': df['Order State'].mode()[0],
        'Order Status': df['Order Status'].mode()[0],
        'Product Name': df['Product Name'].mode()[0],
        'Product Category Id': df['Product Category Id'].mode()[0],
        'Product Price': df['Product Price'].mean(),
        'Order Item Product Price': df['Order Item Product Price'].mean(),
    }, inplace=True)
    print("Handled missing values.")

    # Encode categorical variables
    categorical_cols = ['Category Name', 'Customer City', 'Customer Country', 'Customer Segment',
                        'Department Name', 'Order Region', 'Order State', 'Order Status',
                        'Product Name', 'Product Category Id', 'Product Status', 'Shipping Mode']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    print("Encoded categorical variables.")

    return df

def calculate_demand(df):
    """Calculate demand metrics for each category or product."""
    # Aggregate sales by category
    demand = df.groupby('Category Id').agg({
        'Sales': 'sum',
        'Order Item Quantity': 'sum',
        'Order Item Total': 'sum'
    }).reset_index()

    # Rename columns for clarity
    demand.rename(columns={
        'Sales': 'Total Sales',
        'Order Item Quantity': 'Total Quantity Sold',
        'Order Item Total': 'Total Revenue'
    }, inplace=True)

    # Calculate average price based on total sales and quantity sold
    demand['Average Price'] = demand['Total Revenue'] / demand['Total Quantity Sold']

    print("Calculated demand metrics.")
    return demand

def visualize_demand(demand_df):
    """Visualize demand metrics."""
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Total Sales', y='Category Id', data=demand_df.sort_values('Total Sales', ascending=False).head(10))
    plt.title('Top 10 Categories by Total Sales')
    plt.xlabel('Total Sales')
    plt.ylabel('Category Id')
    plt.tight_layout()
    plt.savefig('../output/top_10_categories_sales.png')
    plt.close()
    print("Saved Top 10 Categories by Total Sales plot.")

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Total Revenue', y='Category Id', data=demand_df.sort_values('Total Revenue', ascending=False).head(10))
    plt.title('Top 10 Categories by Total Revenue')
    plt.xlabel('Total Revenue')
    plt.ylabel('Category Id')
    plt.tight_layout()
    plt.savefig('../output/top_10_categories_revenue.png')
    plt.close()
    print("Saved Top 10 Categories by Total Revenue plot.")

def supply_recommendation(demand_df, threshold_increase=1000, threshold_decrease=500):
    recommendations = demand_df.copy()
    recommendations['Supply Recommendation'] = recommendations['Total Sales'].apply(
        lambda x: 'Increase Supply' if x > threshold_increase else ('Decrease Supply' if x < threshold_decrease else 'Maintain Supply')
    )
    return recommendations[['Category Id', 'Total Sales', 'Supply Recommendation']]

def predict_demand(df):
    """Train a logistic regression model to predict demand based on selected features."""
    global model  # Use the global model variable
    # Define feature columns (adjust based on your data)
    feature_cols = [
        'Order Item Quantity', 'Sales per customer', 'Benefit per order', 'Late_delivery_risk'
    ] + [col for col in df.columns if col.startswith('Category Name_')]  # Adjust based on encoding

    # Create demand target variable
    df['Demand'] = (df['Sales'] > df['Sales'].mean()).astype(int)  # Binary target: 1 if above average sales, else 0

    # Split the dataset into training and testing sets
    X = df[feature_cols]
    y = df['Demand']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)

def predict_manual_input():
    """Predict demand based on manual input from the user."""
    global model  # Use the global model variable
    if model is None:
        print("Model is not trained yet. Please run the training process first.")
        return

    print("Enter the following details for prediction:")
    order_item_quantity = float(input("Order Item Quantity: "))
    sales_per_customer = float(input("Sales per Customer: "))
    benefit_per_order = float(input("Benefit per Order: "))
    late_delivery_risk = int(input("Late Delivery Risk (0 or 1): "))  # Assuming binary

    # Encode the categorical features (you might need to adjust this depending on your encoding)
    # Here, I'm assuming there are no categorical features since they are already one-hot encoded in preprocessing
    # You can add categorical feature inputs if needed

    # Create a DataFrame for input
    input_data = pd.DataFrame({
        'Order Item Quantity': [order_item_quantity],
        'Sales per customer': [sales_per_customer],
        'Benefit per order': [benefit_per_order],
        'Late_delivery_risk': [late_delivery_risk]
    })

    # Add necessary one-hot encoded columns based on the model's features
    for col in model.feature_names_in_:
        if col not in input_data.columns:
            input_data[col] = 0  # Default to 0 for non-existing features
    input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)  # Ensure correct order

    # Make prediction
    prediction = model.predict(input_data)
    print(f"Predicted Demand (1 = High Demand, 0 = Low Demand): {prediction[0]}")

def main():
    # Define file paths
    data_file = '../data/sales_data.csv'
    output_dir = '../output'
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    df = load_data(data_file)
    if df is None:
        return

    # Preprocess data
    df_processed = preprocess_data(df)

    # Debug: Print columns to check if 'Product Category' exists
    print("Columns after preprocessing:", df_processed.columns)

    # Calculate demand
    demand_df = calculate_demand(df_processed)

    # Save demand metrics to CSV
    demand_df.to_csv('../output1/demand_metrics.csv', index=False)
    print("Saved demand metrics to CSV.")

    # Visualize demand
    visualize_demand(demand_df)

    # Provide supply recommendations
    recommendations = supply_recommendation(demand_df)
    recommendations.to_csv('../output/supply_recommendations.csv', index=False)
    print("Saved supply recommendations to CSV.")

    # Predict demand using logistic regression
    predict_demand(df_processed)

    # Manual prediction
    predict_manual_input()

    print("Demand analysis completed successfully.")

if _name_ == "_main_":
    main()
```
## Output
