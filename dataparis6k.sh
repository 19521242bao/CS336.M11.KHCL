# Paris dataset. Download and move all images to same directory.
wget http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_1.tgz
wget http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_2.tgz
mkdir paris6k_images_tmp
tar -xvzf paris_1.tgz -C paris6k_images_tmp/
tar -xvzf paris_2.tgz -C paris6k_images_tmp/
mkdir paris6k_images
mv paris6k_images_tmp/paris/*/*.jpg paris6k_images/
wget http://cmp.felk.cvut.cz/revisitop/data/datasets/rparis6k/gnd_rparis6k.mat