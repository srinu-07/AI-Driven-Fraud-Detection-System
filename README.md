This project aims to develop an AI model to detect fraudulent transactions in 
real-time using historical transaction data. By applying data preprocessing 
techniques and feature engineering, we prepare the dataset for model training. 
We utilize a machine learning algorithm to classify transactions as fraudulent 
or non-fraudulent. The model is evaluated using metrics such as precision, 
recall, and AUC-ROC to ensure its effectiveness in identifying fraud with 
minimal false positives. The ultimate goal is to create a reliable system that can 
accurately flag suspicious activities and enhance transaction security. 

PROPOSED SOLUTION: 

Data Preprocessing: 

 Label Encoding: Categorical variables are converted into numerical 
values using LabelEncoder so they can be processed by the machine 
learning model.

 Handling Imbalanced Data: The SMOTE (Synthetic Minority Over
sampling Technique) is applied to create synthetic samples of the minority 
class, balancing the dataset to improve the model's performance on the 
minority class (fraudulent transactions). 


Model Training: 

 Train-Test Split: The balanced dataset is split into training and testing sets 
using train_test_split to ensure that the model is evaluated on unseen data. 

 Random Forest Classifier: A RandomForestClassifier is trained on the 
training set. This algorithm uses an ensemble of decision trees to make 
predictions, providing robust results due to its ability to handle both 
classification and regression tasks. 


Model Evaluation: 

 Evaluation Metrics: The model's performance is evaluated using metrics 
such as accuracy, precision, recall, and a detailed classification report. 
These metrics help assess how well the model is identifying fraudulent 
transactions. 

 ROC Curve: The ROC (Receiver Operating Characteristic) curve is 
plotted, and the AUC (Area Under the Curve) is calculated to visualize 
the trade-off between the true positive rate and false positive rate, giving 
insight into the model's ability to distinguish between classes. 


Model Saving: 

 Saving the Model: The trained model is saved as a .pkl (pickle) file using 
pickle.dump, allowing it to be loaded and used later without retraining. 

Visualization:  

 Plotting ROC Curve: The ROC curve is plotted using Matplotlib to 
provide a graphical representation of the model's performance, 
particularly in terms of its trade-offs between sensitivity and specificity.

Project Files

The full project files can be accessed at this Google Drive Link. https://drive.google.com/drive/u/1/folders/11n5SLA1vY2WGa3h3L1nlzNAs15-Uj89T

Prerequisites

Before running the project, make sure the following dependencies are installed:

Python 3.x

scikit-learn

pandas

numpy

matplotlib

pickle

imbalanced-learn (for SMOTE)


Usage Instructions

Data Preprocessing: Ensure that the dataset is prepared by applying label encoding and SMOTE to balance the classes.

Train the Model: Run the script to train the Random Forest Classifier on the balanced dataset.

Evaluate the Model: After training, evaluate the model using accuracy, precision, recall, and the ROC curve.

Save and Load the Model: The trained model is saved as a .pkl file and can be reloaded for later use without retraining.

Visualization: Use Matplotlib to plot the ROC curve and visualize the trade-offs between false positives and true positives.

Evaluation

The model's effectiveness is measured based on:

Precision: How many detected frauds are actually frauds.

Recall: How well the model identifies all frauds.

AUC-ROC: A comprehensive metric to evaluate the model's trade-offs between sensitivity and specificity.

Future Enhancements

Feature Engineering: Investigate additional features to improve model accuracy.

Real-time Detection: Integrate the model into a real-time system that can process and flag fraudulent transactions dynamically.

Model Optimization: Tune hyperparameters or explore other algorithms (e.g., XGBoost, LightGBM) for potential performance improvements.


