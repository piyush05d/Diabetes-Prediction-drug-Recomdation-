from setuptools import setup, find_packages

def get_requirements(file_path: str):
    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if "-e ." in requirements:
            requirements.remove("-e .")
    return requirements

setup(
    name="best_drug_detection_for_diabetes",
    version="0.1.0",
    author="Kumar Gaurav",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
