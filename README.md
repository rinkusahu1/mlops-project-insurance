# mlops-project-insurance
Insurance price prediction

JUpyter notebook
tab to get autocomplete

shift+tab inside function to get signature info

Installation and Enabling Conda 
    conda activate exp-tracking-env



1. Install AWS client on linux
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

2. Configure Access Key

aws configure

3. List s3 buckets

aws s3 ls

4.  Start mlflow  server
mlflow server  --backend-store-uri sqlite:///mlflow.db  --default-artifact-root s3://medical-insurance-pp-artifacts

5. Modify notebook for this

