sudo amazon-linux-extras enable epel
sudo yum install -y glibc libstdc++ gcc gcc-c++ make
sudo amazon-linux-extras enable corretto8
sudo yum install -y glibc

sudo yum install -y epel-release

#conda env create -f environment.yml

sudo yum install -y tesseract
