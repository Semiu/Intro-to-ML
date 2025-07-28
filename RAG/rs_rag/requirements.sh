# sudo yum update --yes

# sudo sh -c 'yum install epel-release' -y
# sudo sh -c 'yum install tesseract' -y

conda install -c conda-forge ctransformers -y

sudo yum install poppler-utils -y

# sudo yum install pandoc

# sudo yum install tesseract-ocr -y

sudo yum install "unstructured[all-docs]"
sudo yum install "unstructured[s3]"
sudo yum install "unstructured[rtf]"
