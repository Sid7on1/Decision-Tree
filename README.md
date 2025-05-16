# *****Decision-Tree*****
This project predicts the likelihood of heart disease using a Decision Tree Classifier. It processes clinical data such as age, cholesterol, and blood pressure to identify potential heart conditions. The model provides accurate and interpretable predictions for early diagnosis.
❤️ Heart Disease Prediction using Decision Tree

📌 Project Overview

This project uses a Decision Tree Classifier to predict the presence of heart disease based on patient health metrics such as age, cholesterol levels, chest pain type, blood pressure, and more. The goal is to provide an interpretable and reliable machine learning model that assists with early detection of cardiovascular issues.

🧠 Code and logic written by the user and refined by AI.

⸻

🔍 Problem Statement

Heart disease remains one of the leading causes of death globally. Early detection can save lives. This project leverages supervised machine learning (Decision Trees) to classify whether an individual is likely to have heart disease using medical attributes from the UCI Heart Disease dataset.

⸻

🧪 Dataset

The dataset used is heart.csv from the UCI Machine Learning Repository. It includes the following features:
	•	age
	•	sex
	•	cp (chest pain type)
	•	trestbps (resting blood pressure)
	•	chol (serum cholesterol)
	•	fbs (fasting blood sugar)
	•	restecg (resting ECG)
	•	thalach (max heart rate)
	•	exang (exercise-induced angina)
	•	oldpeak (ST depression)
	•	slope, ca, thal
	•	target (0 = no disease, 1 = disease)

⸻

🧠 How Decision Tree Works

A Decision Tree mimics human decision-making. It recursively splits the dataset based on feature values to maximize class separation. The algorithm chooses splits using criteria like Gini Impurity or Information Gain (Entropy). Each node represents a decision, and the tree grows until it fully classifies the dataset or meets stopping conditions (e.g., max depth or min samples).

Why Decision Tree?
	•	Interpretable & explainable
	•	Handles both numerical and categorical data
	•	Low preprocessing requirement

⸻

⚙️ Technologies Used
	•	Python 3
	•	Pandas
	•	NumPy
	•	Scikit-learn
	•	Matplotlib / Seaborn (optional for EDA)
