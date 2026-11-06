# Hotel Booking Cancellation Prediction

## Project Overview
This project aims to predict whether a hotel booking will be canceled using Machine Learning techniques.

The application allows users to explore hotel booking data and predict cancellation probabilities based on booking information.

## Business Problem
Hotel booking cancellations can lead to revenue loss and operational challenges.

This project helps hotels identify bookings that are more likely to be canceled, allowing them to improve planning and reduce potential losses.

## Dataset
The project uses the Hotel Booking Demand dataset from Kaggle.

Dataset: [Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Matplotlib
- Seaborn

## Machine Learning Models
The project explores several classification algorithms:

- Logistic Regression
- Decision Tree
- Random Forest

The trained model is saved and used for prediction through the Streamlit application.

## Project Features
- Hotel booking data analysis
- Data preprocessing
- Machine Learning model training
- Model evaluation and comparison
- Interactive prediction interface
- Cancellation prediction based on booking details

## Project Structure

```text
hotel-booking-cancellation/
│
├── app.py
├── hotel_data.csv
├── hotel_model.pkl
├── model_results.csv
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
```

Navigate to the project directory:

```bash
cd hotel-booking-cancellation
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Open the local URL provided by Streamlit in your browser.

## Deployment
The application is intended to be deployed using Streamlit Community Cloud.

## Future Improvements
- Improve model performance through hyperparameter tuning.
- Add more interactive visualizations.
- Improve the user interface.
- Deploy the application publicly.
- Add business insights and recommendations.
