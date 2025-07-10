### Objective
Demonstrate MLOps Experiment tracking using MLFlow

### MLFlow vs DVC

1. DVC (Data Version Control) and MLflow are both popular open-source tools for machine learning workflows, but they have different primary focuses. 
2. DVC excels at data and model versioning, while MLflow is designed for end-to-end ML lifecycle management, including experiment tracking, model registry, and deployment. 
3. They can also be used together to create a comprehensive MLOps pipeline. 

### DVC (Data Version Control):
- Core Function:
    DVC specializes in versioning large datasets, models, and other large files associated with machine learning projects. 
- Key Features:
    - Data and Model Versioning: DVC uses Git-like commands to manage versions of data and models, storing metadata (references to the actual files) in Git repositories. 
    - Pipeline Management: DVC allows defining and tracking dependencies between data processing and model training stages, enabling reproducible workflows. 
    - Data Sharing: DVC facilitates sharing data and model files between team members and systems using various storage backends. 
    - Git Integration: DVC seamlessly integrates with Git, allowing you to manage your data and pipelines alongside your code. 
- Strengths:
    DVC is particularly useful for managing the data and model versions, ensuring reproducibility and facilitating collaboration on large datasets. 
- Example Use Cases:
    Tracking different versions of a training dataset, managing different model versions, and defining reproducible data pipelines. 

### MLflow:
- Core Function:
    MLflow provides a platform for managing the entire machine learning lifecycle, including experiment tracking, model registry, and deployment. 
- Key Features:
    - Experiment Tracking: MLflow allows logging parameters, metrics, and artifacts for each experiment, making it easy to track and compare results. 
    - Model Registry: MLflow provides a centralized registry for managing and versioning models, enabling easy deployment and monitoring. 
    - Model Serving: MLflow offers tools for deploying models to different platforms and monitoring their performance. 
    - Integration: MLflow integrates with various cloud platforms (AWS, Azure, GCP) and machine learning libraries (like scikit-learn). 
- Strengths:
    MLflow is well-suited for managing the entire ML lifecycle, especially experiment tracking, model versioning, and deployment. 
- Example Use Cases:
    Tracking hyperparameters of different model versions, logging model performance metrics, and deploying models to production. 

### What Model attributes can be logged using MLFlow?
MLflow is a powerful tool for tracking and managing machine learning experiments. Here’s a list of things that can be tracked/logged using MLflow, which you can include in your tutorial documentation:

#### 1. **Metrics:**
   - **Accuracy**: Track model accuracy over different runs.
   - **Loss**: Log training and validation loss during the training process.
   - **Precision, Recall, F1-Score**: Log evaluation metrics for classification tasks.
   - **AUC (Area Under Curve)**: Track AUC for classification models.
   - **Custom Metrics**: Any numeric value can be logged as a custom metric (e.g., RMSE, MAE).

#### 2. **Parameters:**
   - **Model Hyperparameters**: Log values such as learning rate, number of trees, max depth, etc.
   - **Data Processing Parameters**: Track parameters used in data preprocessing, such as the ratio of train-test split or feature selection criteria.
   - **Feature Engineering**: Log any parameters related to feature extraction or engineering.

#### 3. **Artifacts:**
   - **Trained Models**: Save and version models for easy retrieval and comparison.
   - **Model Summaries**: Log model summaries or architecture details.
   - **Confusion Matrices**: Save visualizations of confusion matrices.
   - **ROC Curves**: Log Receiver Operating Characteristic curves.
   - **Plots**: Save any custom plots like loss curves, feature importances, etc.
   - **Input Data**: Log the datasets used in training and testing.
   - **Scripts & Notebooks**: Save code files or Jupyter notebooks used in the experiment.
   - **Environment Files**: Track environment files like `requirements.txt` or `conda.yaml` to ensure reproducibility.

#### 4. **Models:**
   - **Pickled Models**: Log models in a serialized format that can be reloaded later.
   - **ONNX Models**: Log models in the ONNX format for cross-platform usage.
   - **Custom Models**: Log custom models using MLflow’s model interface.

#### 5. **Tags:**
   - **Run Tags**: Tag your experiments with metadata like author name, experiment description, or model type.
   - **Environment Tags**: Tag with environment-specific details like `gpu` or `cloud_provider`.

#### 6. **Source Code:**
   - **Scripts**: Track the script or notebook used in the experiment.
   - **Git Commit**: Log the Git commit hash to link the experiment with a specific version of the code.
   - **Dependencies**: Track the exact version of libraries and dependencies used.

#### 7. **Logging Inputs and Outputs:**
   - **Training Data**: Log the training data used in the experiment.
   - **Test Data**: Log the test or validation datasets.
   - **Inference Outputs**: Track the predictions or outputs of the model on a test set.

#### 8. **Custom Logging:**
   - **Custom Objects**: Log any Python object or file type as a custom artifact.
   - **Custom Functions**: Track custom functions or methods used within the experiment.

#### 9. **Model Registry:**
   - **Model Versioning**: Track different versions of models and their lifecycle stages (e.g., `Staging`, `Production`).
   - **Model Deployment**: Manage and track the deployment status of different models.

#### 10. **Run and Experiment Details:**
   - **Run ID**: Each run is assigned a unique identifier.
   - **Experiment Name**: Group multiple runs under a single experiment name.
   - **Timestamps**: Log start and end times of each run to track duration.

### What is the relation between Experiment and Runs in MLFlow?
In MLflow, an experiment is a container for related runs, while a run represents a single execution of a machine learning model or code. Think of an experiment as a project folder, and runs as individual tasks within that project.
We use experiments to organize your work, and runs to track the details of individual executions of our machine learning code

### How to overcome errors with mlflow.log_model() during Experiment Tracking APIs?
the current version of MLFlow often throws errors while trying to log model. To overcome this error, pip install mlflow==2.2.2 which is stable and should not cause the error

---

### MLFlow autolog feature:

**`mlflow.autolog()`** is a powerful feature in MLflow that automatically logs parameters, metrics, models, and other relevant information during your machine learning training process. However, it's important to know what can and cannot be logged automatically.

#### **Things That Can Be Logged by `mlflow.autolog`:**

1. **Parameters:**
   - Hyperparameters used to train the model, such as `max_depth`, `learning_rate`, `n_estimators`, etc.

2. **Metrics:**
   - Common evaluation metrics like accuracy, precision, recall, and loss values, depending on the model and framework being used.

3. **Model:**
   - The trained model itself is automatically logged.

4. **Artifacts:**
   - Certain artifacts like model summary and plots (e.g., learning curves, confusion matrix) are logged if supported by the framework.

5. **Framework-Specific Information:**
   - Framework-specific details like early stopping criteria in gradient boosting models or deep learning models (e.g., number of epochs, optimizer configuration).

6. **Environment Information:**
   - Details about the environment such as installed libraries and versions.

7. **Training Data and Labels:**
   - Information about the dataset size and sometimes feature information, but not the entire dataset itself.

8. **Automatic Model Signature:**
   - Autologging can infer the input types (signature) of the model and save them along with the model.

#### **Things That Cannot Be Logged by `mlflow.autolog`:**

1. **Custom Metrics:**
   - Metrics not included in the default set for the specific framework (e.g., F1 score if it's not the default metric) will not be logged unless manually specified.

2. **Custom Artifacts:**
   - Custom plots, charts, or files that are not part of the default model training process (e.g., a custom visualization or report).

3. **Preprocessed Data:**
   - The transformed or preprocessed data used during training or testing is not logged unless you manually log it as an artifact.

4. **Intermediate Model States:**
   - Models saved at intermediate stages of training (e.g., after every epoch) are not logged unless explicitly done so.

5. **Complex Model Structures:**
   - If you're using a non-standard or highly customized model structure, `mlflow.autolog` might miss some logging details.

6. **Non-standard Training Loops:**
   - If your training loop is not compatible with the standard loops expected by MLflow (e.g., custom training loops), autologging might not capture everything correctly.

7. **Non-Supported Frameworks:**
   - `mlflow.autolog` does not support all frameworks. If your model is built with a framework that MLflow doesn’t support, autologging won’t work.

8. **Custom Hyperparameter Tuning:**
   - Hyperparameters or configurations that are outside the scope of the framework’s autologging capabilities (e.g., specific settings in a custom grid search).

### Summary:

- **Use Cases:** `mlflow.autolog` is great for quick and convenient logging, especially for standard workflows in supported frameworks.
- **Limitations:** Custom elements, complex structures, and unsupported frameworks require manual logging to capture all relevant details.
