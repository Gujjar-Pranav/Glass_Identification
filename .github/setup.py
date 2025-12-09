from setuptools import setup, find_packages

setup(
    name="glass_ensemble_ml",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "imbalanced-learn",
        "joblib",
    ],
    author="Pranav",
    description="Glass Identification using ensemble learning (Random Forest, Gradient Boosting, Stacking).",
)
