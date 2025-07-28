#!/bin/bash

set -ex

# Update and install all necessary packages at once to reduce redundancy and speed up the process
yum -y update && yum install -y clang libpng-devel libtiff-devel zlib-devel libwebp-devel libjpeg-turbo-devel git-core libtool pkgconfig

# Install leptonica in parallel
wget https://github.com/DanBloomberg/leptonica/releases/download/1.75.1/leptonica-1.75.1.tar.gz
tar -xzvf leptonica-1.75.1.tar.gz
cd leptonica-1.75.1
./configure && make -j$(nproc) && make install
cd ..

# Install autoconf-archive in parallel
wget http://mirror.squ.edu.om/gnu/autoconf-archive/autoconf-archive-2017.09.28.tar.xz
tar -xvf autoconf-archive-2017.09.28.tar.xz
cd autoconf-archive-2017.09.28
./configure && make -j$(nproc) && make install
cp m4/* /usr/share/aclocal/
cd ..

# Install tesseract in parallel
git clone --depth 1 https://github.com/tesseract-ocr/tesseract.git tesseract-ocr
cd tesseract-ocr
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig
./autogen.sh
./configure
make -j$(nproc)
make install
export TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/

# Download and install eng data for Tesseract
wget https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata
mv -v eng.traineddata /usr/local/share/tessdata/