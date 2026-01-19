# ML-Project-SilentWhisper
## Description:
ML-Project-SilentWhisper is a machine learning–based project focused on real-time sign language recognition and translation using computer vision techniques. It leverages hand-tracking, feature extraction and trained ML models to interpret sign gestures from live video input. The project aims to bridge communication gaps between hearing and speech impairment by converting sign language into readable text or speech designed as a learning-driven and practical implementation. It demonstrates the application of AI, computer vision and pattern recognition in assistive technology highlighting real-world impact, accessibility and intelligent between human and computer interaction.

# App Setup:
## All command must write in Command prompt, other wise it causes ERROR.

## Step 1:
### Install Python 3.10 or 3.9
- Download [Python 3.10 Filehippo](https://filehippo.com/download_python/3.10.0/) 

## Step 2: 
### Initialize tailwindcss in cli mode on folder ML-Project-SilentWhisper
- [tailwindcss](https://tailwindcss.com/docs/installation/tailwind-cli)


## Step 3:
### Change Directory to sign_env
```cmd
cd sign_env
```


## Step 4:
### Create Python virtual environment by typing command
```cmd
py -3.10 -m venv venv
```


## Step 5:
### Activate environment script by typing command
```cmd
venv\Scripts\activate
```


## Step 6:
### Install necessary libraries by typing commands
```cmd
pip install --upgrade pip
```
```cmd
pip install mediapipe==0.10.9
```
```cmd
pip install opencv-python
```
```cmd
pip install numpy
```
```cmd
pip install scikit-learn
```
```cmd
pip install flask flask-cors
```


## Step 7:
### Now check the environment are ready or not by typing
```cmd
python -c "import cv2, mediapipe; print('OK')"
```
#### (If console output OK, then you are in right environment other wise install libraries again from Step 6:)


## Step 8:
### run the app by typing
```cmd
python app.py
```


## Step 9:
#### Open index.html from 'src' folder by live server

### Enjoy basic Machine Learning Project!
