# 🩺 Gallstone Risk Prediction – COMEXUS / University of Arizona AI Fellowship

This project was developed as the **final capstone** of the COMEXUS Fellowship at the **University of Arizona**, a program focused on **Artificial Intelligence** and its applications in solving real-world problems.

The goal is to predict the **probability of gallstone presence** using **supervised Machine Learning** (Logistic Regression) trained on **clinical and biochemical data**.

The app was built using [Streamlit](https://streamlit.io/) and allows:
- **Individual predictions** via a web form.
- **Batch predictions** by uploading a CSV file.
- **Model evaluation metrics** if the CSV contains ground truth labels.

---

## 📂 Project Structure
ml-gallstones-app/
│
├── app.py # Streamlit app
├── train.py # Script to train Logistic Regression model
├── features.py # Feature engineering & preprocessing
├── model.joblib # Trained model (Logistic Regression)
├── requirements.txt # Project dependencies
├── data/
│ └── .gitkeep # Empty marker to keep folder in repo
└── README.md

---

## 🚀 Live Demo

You can try the app here: **[Live Streamlit App](https://YOUR-APP-URL.streamlit.app)**

---

## 🧠 Machine Learning Pipeline

1. **Problem Definition**  
   Predict gallstone status (binary classification) from clinical and biochemical data.

2. **Data Preprocessing**  
   - Decimal conversion (comma to dot).  
   - Missing value imputation (mean).  
   - Outlier handling (only clear errors removed).  
   - Standardization (z-score scaling).

3. **Feature Engineering**  
   - Ratios: `ALT/AST`, `Cholesterol/HDL`, `LDL/HDL`, `Triglyceride/HDL`.  
   - Binary health flags: Obese, High Cholesterol, Hyperglycemia, Low Vitamin D.  
   - Age groups (one-hot encoding).  
   - Interaction features.

4. **Model Selection**  
   - Algorithm: **Logistic Regression** (L1 regularization, `liblinear` solver).  
   - Cross-validation: 5-fold stratified.  
   - Threshold tuning to maximize F1-score.

5. **Deployment**  
   - Streamlit web app.  
   - Supports individual and batch predictions.  
   - Displays metrics if true labels are provided.

---

## 🖥 Running Locally

### 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/ml-gallstones-app.git
cd ml-gallstones-app

2. Create virtual environment
python -m venv .venv
# Activate (Windows)
.venv\Scripts\Activate.ps1
# Activate (macOS/Linux)
source .venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

4. (Optional) Retrain the model

If you want to retrain the model using your own dataset, place it in data/ and run:

python train.py


This will overwrite model.joblib.

5. Run the app
streamlit run app.py

📤 Deployment on Streamlit Cloud

Push this repository to GitHub.

Go to share.streamlit.io and log in with GitHub.

Create a New app:

Repository: YOUR_USERNAME/ml-gallstones-app

Branch: main

Main file path: app.py

Deploy and get your public URL.

📊 Example Inputs
High risk case (likely positive):

Age: 62, BMI: 33.8, VFA: 160.0, Glucose: 145, HDL: 35, LDL: 185, TG: 260, Vitamin D: 14, ALT: 78, AST: 52

Low risk case (likely negative):

Age: 28, BMI: 22.1, VFA: 35.0, Glucose: 88, HDL: 58, LDL: 95, TG: 90, Vitamin D: 32, ALT: 22, AST: 20

📜 License

This project is for educational and demonstration purposes only.
It is not a substitute for professional medical diagnosis or advice.

👩‍💻 Author

Developed by [Your Name] as part of the COMEXUS / University of Arizona AI Fellowship.
