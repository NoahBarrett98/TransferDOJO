import os
from setuptools import setup

setup(
    name="TransferDOJO",
    version="0.1.1",
    description="Transfer Learning Experiments",
    author="Noah Barrett",
    include_package_data=True,
    zip_safe=False,
    packages=['TransferDOJO'],
    install_requires=[
        "click==7.1.2",
        "click_logging",
        "efficientnet-pytorch",
        "matplotlib",
        "numpy",
        "opencv-python==4.5.2.52",
        "optuna==2.7.0",
        "pandas==1.1.3",
        "Pillow",
        "protobuf==3.15.8",
        "scikit-learn==0.24.2",
        "scipy",
        "tensorboard==2.3.0",
        "tensorboardX==2.1",
        "torch==1.8.1",
        "seaborn",
        "torchvision",
        "tqdm==4.50.2",
        "mlflow",
        "ReprDynamics"
    ],
    entry_points={
        "console_scripts": [
            "train_model = TransferDOJO.main:train_model",
            "train_bootstrap = TransferDOJO.main:train_bootstrap",
            "hparam_search = TransferDOJO.main:hparam_search_wrapper",
            "evaluate_tuned_models = TransferDOJO.evaluation:evaluate_tuned_models",
            "evaluate_model = TransferDOJO.evaluation:evaluate_model",
        ]
    },
)
