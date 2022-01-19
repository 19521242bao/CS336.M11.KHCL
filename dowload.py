# Don't want to crop query img?
# Just download prepared query img
# Enabled these below line and run then move directly to run classifying step

import gdown

modelUrl = 'https://drive.google.com/uc?id=1hUff66nwWyhBf1UG-UrcngYHPSXgcGUs' #URL cố định dùng để download.
output = '/content/eval/query_img/query_img.zip' 
gdown.download(modelUrl, output, quiet=False)

!unzip -o '/content/eval/query_img/query_img.zip' -d '/content/eval'
!rm -r '/content/eval/query_img/query_img.zip'