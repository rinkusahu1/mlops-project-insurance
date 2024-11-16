
# Installation Guide

## Prerequisites  
Before you begin the installation process, ensure that the following software and tools are installed on your system:

1. **Python 3.x**  
   - This project requires Python version 3.10 or higher.  
   - Download from [Python.org](https://www.python.org/downloads/).

2. **Git**  
   - Git is needed for version control and cloning the repository.  
   - Install from [Git Downloads](https://git-scm.com/downloads).

3. **Package Manager (optional)**  
   - **Conda** (recommended for isolated environments)  
     Install [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
   - **Pip**  
     Pip comes with Python but ensure it’s up to date:  
     ```bash
     python -m pip install --upgrade pip
     ```

4. **Docker** (optional, for containerized environments)  
   - Docker is recommended if you want to run the project in a container.  
   - Download Docker from [Docker.com](https://www.docker.com/get-started).

---

## Operating System Requirements  
The project is compatible with the following operating systems:
- **Linux**  



## Installation Steps

### Conda env setup
#### Step 1: Create Conda environment
- conda create -n  medical-price-prediction  python=3.10

#### Step 2: Activat Conda
- conda activate exp-tracking-env

#### Step 3: Install required packages
- Go to project base directory
- pip install -r requirements.txt


###  Mlflow Setup

#### Start mlflow server
- `mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root s3://medical-insurance-pp-artifacts`
- models are deployed on s3 bucket. 


### Mage Trainig Pipeline
- Go to PROJECTBASEFOLDER/mlops/scripts.
- Run `sh start.sh` command.


### Following Coding Standard

#### Install pre-commit hook
- Ensure that unit tests pass and code is properly formatted before every commit.
- Run `pre-commit install` command.

### Local Integration Testing

- Go to PROJECTBASEFOLDER/best-practice/integration_test.
- Run `sh run.sh` command.

### Monitoring best-practice/monitoring/Insurance Price Prediction Monitoring.ipynb
-  Go to PROJECTBASEFOLDER/best-practice/monitoring
- Open `Insurance Price Prediction Monitoring.ipynb` file in notebook

### Evidently dashboard
-  Go to PROJECTBASEFOLDER/best-practice/monitoring
- Run `evidently ui` command.

### CI/CD pipeline 
- CI pipeline file .github/workflows/ci-tests.yml

- CD pipeline file .github/workflows/cd-deploy.yml, update MODEL_BUCKET_DEV and RUN_ID in `cd-deploy.yml` as per your usage.

#### Deployment Testing
- Go to PROJECTBASEFOLDER/best-practice/scripts/
- Run  `sh test-cloud-e2e.sh` command, update `KINESIS_STREAM_OUTPUT`,`KINESIS_STREAM_INPUT` variable based on environment deployment










