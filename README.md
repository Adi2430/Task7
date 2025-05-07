# Task7

 Task 7: Support Vector Machines (SVM)
Objective: Apply SVMs for linear and non-linear classification using the Breast Cancer dataset.

Step 1: Load and Prepare the Dataset
I used a Breast Cancer dataset containing diagnostic information about tumors.
The dataset included features like radius_mean, texture_mean, and so on, and a target column called diagnosis (with values 'M' for malignant and 'B' for benign).
The non-numeric labels in the 'diagnosis' column were converted to numeric values:
'M' → 1 (malignant)
'B' → 0 (benign)
The unnecessary id column was dropped to focus only on relevant features.

Step 2: Preprocessing
I scaled the feature data using StandardScaler from Scikit-learn. This ensures that all features contribute equally to the SVM model.
I split the dataset into training and testing sets (80% train, 20% test), allowing us to train the model and evaluate its performance later on unseen data.

Step 3: Train SVM Models (Linear & RBF Kernel)
I trained two SVM classifiers:
Linear SVM: For cases where data is linearly separable.
RBF (Radial Basis Function) SVM: For handling non-linear decision boundaries.
Both models were trained on the scaled training data.
After training, I used the test set to predict and evaluate the model's performance using classification reports.

Step 4: Visualize Decision Boundaries using PCA
Since the dataset is high-dimensional (30 features), I applied Principal Component Analysis (PCA) to reduce the feature space to 2 dimensions for visualization.
I plotted the decision boundaries of both Linear and RBF SVM models using matplotlib, showing how they separate the two classes in 2D space.

Step 5: Hyperparameter Tuning
I used GridSearchCV to find the best values for key SVM hyperparameters:
C: Controls margin softness (penalty for misclassification).
gamma: Defines influence of each data point (for RBF kernel).
The grid search tested multiple combinations of C and gamma using 5-fold cross-validation.
The best model parameters were printed, along with the best cross-validation score.
