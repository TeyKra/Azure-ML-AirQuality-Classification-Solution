# Azure Machine Learning Project: Air Quality Classification

## Project Overview
This project addresses a classification problem related to air quality. It demonstrates the complete end-to-end process of building, optimizing, deploying, and monitoring a machine learning model using Azure Machine Learning. The project involves creating a robust machine learning pipeline, utilizing tools such as Optuna for hyperparameter tuning, MLflow for experiment tracking, and deploying the model as a REST API with an optimum front-end interface for real-time predictions.

---

## Key Features
1. **Data Exploration and Preprocessing**
   - Dataset sourced from public platforms like Kaggle or UCI.
   - Data preprocessing: handling missing values, encoding categorical variables, scaling numerical features.

2. **Model Training and Optimization**
   - Built initial model using algorithms such as Random Forest and XGBoost.
   - Hyperparameter tuning with Optuna or Azure ML HyperDrive.
   - Experiment tracking using MLflow.

3. **Model Deployment**
   - Deployed as a REST API endpoint on Azure Machine Learning.
   - Developed a scoring script (`score.py`) for endpoint inference logic.

4. **Front-End Interface**
   - Created an interactive application using Streamlit (`app.py`) for user input and real-time predictions.

5. **Monitoring and Evaluation**
   - Monitored model performance, tracked latency, and logged metrics using Azure tools.

---

## Repository Contents
- **`data_exploration_to_deployment.ipynb`**: Step-by-step Jupyter Notebook for data exploration, preprocessing, model training, and deployment.
- **`data_exploration_to_deployment.py`**: End-to-end pipeline script for setting up the environment, training the model, and deploying it.
- **`pipeline_logs.txt`**: Execution logs from the Azure compute instance.
- **`score.py`**: Scoring script for handling model inference requests at the deployed endpoint.
- **`app.py`**: Streamlit app for real-time user interaction and prediction visualization.
- **`test_input.json`**: Sample input JSON for testing the deployed endpoint.
- **`requirements.txt`**: Installation of the Python dependencies (libraries) used. 
- **`Azure Machine Learning Air Quality Classification Project.pdf`**: Detailed project report documenting all phases, challenges, and results.

---

## Getting Started
### Prerequisites
- Azure subscription with Azure Machine Learning resources configured.
- Python 3.8 or later with the required libraries installed (`requirements.txt` provided in the project).

### Setup and Execution
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the pipeline script:
   ```bash
   python data_exploration_to_deployment.py
   ```

4. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

5. Test the deployed model endpoint using `test_input.json`.

---

This project was developed from scratch under time constraints. As a result, it is not perfect and may contain errors or potential improvements. If you find it useful, feel free to create an issue to share your feedback with me.

Thank you!
