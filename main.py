import kagglehub

# Download latest version
path = kagglehub.dataset_download("juniorbueno/neural-networks-homer-and-bart-classification")

print("Path to dataset files:", path)