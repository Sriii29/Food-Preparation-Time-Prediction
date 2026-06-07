# Food-Preparation-Time-Prediction
This project predicts food order preparation time using Machine Learning and NLP. It analyzes order details, timing, weather, events, and kitchen activity to estimate how long food will take. Gradient Boosting achieved 92% accuracy.

The system looks at things like what food people have ordered, when they ordered it, the weather, special events and how busy the kitchen is. It uses all this information to estimate how long the food will take to prepare. 

The project takes the information about the food. Turns it into numbers that the computer can understand. It does this using TF-IDF Vectorisation. Then it combines this information with data from the restaurant to train the model. The project tried out a few algorithms to see which one worked best. These algorithms were Linear Regression, Random Forest, Gradient Boosting and Quantile Gradient Boosting. 

# Key Features: 
• Food preparation time prediction 
• The project uses Natural Language Processing to understand the food items. It does this with TF-IDF 
• The project can also give you a range of times when the food will be ready so you can plan ahead (P50 and P90 predictions) 
• The backend of the project is powered by FastAPI 
• The project is designed to work in a restaurant or in the cloud

## Technologies Used
- Python
- Scikit-Learn
- FastAPI
- Pandas
- NumPy
- TF-IDF Vectorizer (NLP)
- FastAPI
- Flask
- HTML/CSS
- Joblib
- Uvicorn

## Machine Learning Models Used
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- Quantile Gradient Boosting 
- Regressor (P50 & P90 Prediction)

## Dataset
Custom food preparation dataset used for training and testing.

## Project Structure
ClientUI/
FastApi/
data_generator.py
project_code.py
food_prep_dataset.csv

## Future Improvements
- Deploy on cloud
- Improve model accuracy
- Add real-time predictions
