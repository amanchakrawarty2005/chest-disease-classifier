import importlib

packages = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scikit-learn": "sklearn",
    "scipy": "scipy",
    "tensorflow": "tensorflow",
    "keras": "keras",
    "opencv-python": "cv2",
    "Pillow": "PIL",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "streamlit": "streamlit",
    "requests": "requests",
    "python-multipart": "multipart",
    "pydantic": "pydantic",
    "mlflow": "mlflow",
    "pytest": "pytest",
    "pytest-cov": "pytest_cov",
    "python-dotenv": "dotenv",
    "tqdm": "tqdm",
    "pyyaml": "yaml"
}

print("=" * 50)
print("Checking Installed Packages")
print("=" * 50)

for package_name, import_name in packages.items():
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, "__version__", "Version not found")
        print(f"✅ {package_name:<20} Installed | Version: {version}")
    except ImportError:
        print(f"❌ {package_name:<20} NOT Installed")

print("=" * 50)