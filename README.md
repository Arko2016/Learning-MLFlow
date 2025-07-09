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

