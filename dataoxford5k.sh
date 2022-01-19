mkdir data
wget http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images.tgz
mkdir oxford5k_images
mkdir oxford5k_features
tar -xvzf oxbuild_images.tgz -C oxford5k_images/