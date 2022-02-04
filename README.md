# CS336.M11.KHCL 
Information Retrieval Final Project  
Supported features:
- Retrieve methods DELF, CNN (ResNet), CNNIRPYTORCH are available on web app.
- Particular object search: allow user to choose a specific area of the query image.

# Contributors
|Name               | Email                  | Github profile                                |
|-------------------|------------------------|-----------------------------------------------|
|Phạm Ngọc Dương    | 19521412@gm.uit.edu.vn | [pnd280](https://github.com/pnd280)           |
|Lương Phạm Bảo     | 19521242@gm.uit.edu.vn | [19521242bao](https://github.com/19521242bao) |
|Nguyễn Gia Thống   | 19520993@gm.uit.edu.vn | -                                             |

# Repo structure 
Overview
```
.
├── ...
├── retrieval.py
├── app.py
├── static
│   ├── features
│   ├── images
│   ├── query
│   └── uploads
└── templates
```
Files:
- **./app.py**: flask web server.
- **./retrieval.py**: retrieve images (standalone).

Folders:
- **./static/**: contains 2 datasets including features and 256x256 resized version, uploaded query images.

# Usage
You can clone this repo and install datasets separately or you can download the whole compressed file that we have already downloaded and structured.

## Install datasets separately
Download the compressed file [here](https://drive.google.com/file/d/1ZVeavzf2ohoQrhxA17rBQ_2HacANbBGP/view?usp=sharing) and we require user an extra structuring step, **./static** should look like this:

```
static
├── features
│   ├── feature
│   ├── feature_oxford
│   ├── feature_oxford_2
│   └── feature_paris
└── images
    ├── database_oxford
    ├── database_paris
    ├── resized_oxford
    └── resized_paris

```

## Install the whole structured folder 
This compressed file requires no further resource downloads or structuring.  
[Download](https://drive.google.com/file/d/1F5ddOemdecSETRYEq22DN30xn4Ex9jNY/view?usp=sharing)
## Install required libraries
**./requirements.txt**  
and
```sh
pip install torch===1.5.0 torchvision===0.6.0 -f https://download.pytorch.org/whl/torch_stable.html
```
## Start web server
**cd** to top level folder
```sh
flask run
```
or 
```sh
python app.py
```

If correctly configured, server will be accessible at http://127.0.0.1:5000.

## Query image without starting web server
As we have mentioned above, we will use **./retrieval.py**. We're too lazy too add some extra lines of code to implement command line call, so you have to directly run the file itself.  

Example:
```py
retrieval_image("<query_img_full_path>", "<method>", "<dataset>")
```

# Demo
Yes, we do support image cropping, directly click into the image to crop it!  
Of course the cropped part will be the new query.
![Demo](https://github.com/19521242bao/CS336_M11.KHCL/blob/Web-app/demo/CNN_all_souls_1.png?raw=true)

## Video
[![Demo](https://github.com/19521242bao/CS336_M11.KHCL/blob/Web-app/demo/YTThumbnail.png?raw=true)](https://www.youtube.com/watch?v=FNoluBtsCA0)
