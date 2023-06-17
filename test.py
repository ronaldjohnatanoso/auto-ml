from pycaret.classification import *
from pycaret.datasets import *
import pandas as pd

data = get_data('juice')

best_model = None
best_metric = 0.0

# Loop through each feature
for feature in data.columns:
    print(f"Comparing models with {feature} as the target variable")

    # Get the value counts for each class in the target feature
    class_counts = data[feature].value_counts()

    # Check if any class has only one instance
    if any(class_counts == 1):
        print("Skipping", feature)
        continue

    # Set up the data with the current feature as the target
    s = setup(data, target=feature, session_id=28)

    # Compare the models
    model = compare_models()

    # Evaluate the model using cross-validation
    cv_results = evaluate_model(model, fold=10, round=4, verbose=False)

    # Select the best model based on the evaluation metric
    metric_value = cv_results['Accuracy'].mean()  # Choose the metric of your choice
    if metric_value > best_metric:
        best_metric = metric_value
        best_model = model

    # Print the evaluation metrics for comparison
    print(cv_results)
    print()

# Print the overall best model
print("Overall Best Model:")
print(best_model)
