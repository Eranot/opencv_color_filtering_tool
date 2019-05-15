# OpenCV color filtering tool

![Banner](https://i.imgur.com/c7q5B5x.png "OpenCV color filtering tool")

A tool to easily find lower and upper bounds to color filtering in RGB, HSL and HSV working with OpenCV in Python3

## Installation

Make sure you have OpenCV and Numpy

```bash
sudo pip3 install opencv-python
sudo pip3 install numpy
```

Clone this project

```bash
git clone https://github.com/Eranot/opencv_color_filtering_tool.git
```

(or download it as zip)

## Usage

Execute the script for a image:

```bash
python3 opencv_color_filtering_tool.py -f [image_file]
```

Execute the script for a video:

```bash
python3 opencv_color_filtering_tool.py -f [video_file] --video
```

The parameter -r can be used to resize the image/video where 0.5 makes it half the size and 2.0 doubles it

```bash
python3 opencv_color_filtering_tool.py -f [video_file] --video -r 0.5
```

The parameter --help can be used to get more information

```bash
python3 opencv_color_filtering_tool.py --help
```
