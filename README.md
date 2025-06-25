# 🎓 Student Exam Performance Indicator

An **End-to-End Machine Learning Project** that predicts a student's math score based on demographic and academic information.  
This project demonstrates the complete ML workflow: data ingestion, preprocessing, model training, evaluation, and deployment with a modern web interface.

---

## 🌐 Live Demo

- 🚀 [Live App on Render](https://maths-score-prediction-lpvz.onrender.com)
- 🐙 [GitHub Repository](https://github.com/mayank-kumar03/Maths_score_prediction)

---

## ✨ Features

- 📥 **Data Ingestion:** Automated reading and splitting of raw data.
- 🧹 **Data Transformation:** Robust preprocessing pipelines for numerical and categorical features.
- 🤖 **Model Training:** Multiple regression models with hyperparameter tuning (`GridSearchCV`).
- 📈 **Model Evaluation:** R² score and model comparison report.
- 💾 **Model Persistence:** Save and load models and preprocessors using `dill`.
- 🌐 **Web Application:** User-friendly Flask app with Bootstrap styling and animations.
- ⚡ **Real-time Prediction:** Instantly predict math scores based on user input.
- 🐞 **Exception Handling & Logging:** Custom exception classes and detailed logging for debugging and traceability.
- 🧩 **Modular Codebase:** Clean, maintainable, and scalable Python modules.

---

## 🛠️ Tech Stack

- 🐍 **Python 3.8+**
- 🧮 **Pandas, NumPy, Scikit-learn**
- 🌲 **XGBoost, CatBoost**
- 🌐 **Flask (Web Framework)**
- 🎨 **Bootstrap 5 (UI Styling)**
- 🥒 **dill (Serialization)**
- 📋 **Logging**

---

## 📁 Project Structure

```
Maths_score_prediction/
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
│
├── templates/
│   ├── home.html
│   └── index.html
│
├── notebook/
│   └── data/
│       └── stud.csv
│
├── app.py
├── README.md
└── requirements.txt
```

---

## 🚀 How to Run

1. **Clone the repository:**
   ```sh
   git clone https://github.com/mayank-kumar03/Maths_score_prediction
   cd Maths_score_prediction
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Run the pipeline (optional, for training):**
   ```sh
   python -m src.components.data_ingestion
   ```

4. **Start the web app:**
   ```sh
   python app.py
   ```
   Visit [http://localhost:5000](http://localhost:5000) in your browser.

---

## 📊 Demo

- Enter student details on the home page.
- Click **Predict your Maths Score**.
- Instantly see the predicted math score with a modern, animated UI.

---

## 👨‍💻 Author

- Mayank Kumar

---

## 📢 License

This project is for educational purposes.

---

**Enjoy predicting student performance with a complete, production-ready ML pipeline!**
