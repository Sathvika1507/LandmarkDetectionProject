# 🗺️ Landmark Detection using TensorFlow

This internship project demonstrates facial landmark detection using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The model is trained to predict key facial points from grayscale images.

LandmarkDetectionProject/
├── train_model.py               # Trains the CNN and saves landmark_model.h5
├── predict.py                   # Loads the model and predicts landmarks on test.jpg
├── test.jpg                     # Sample image for testing prediction
├── landmarks_dataset/           # Dataset folder
│   ├── images/
│   │   └── sample.jpg           # Sample training image
│   └── labels.csv              # CSV file containing image names and landmark coordinates
├── .gitignore                   # (Optional) To ignore large model files like .h5
└── README.md                    # Project documentation (this file)


## 🧠 Model Details

- Input: 96x96 grayscale image
- Architecture: Conv2D → MaxPooling → Flatten → Dense
- Output: 30 facial landmark coordinates (15 x, y pairs)
- Loss: Mean Squared Error (MSE)
- Optimizer: Adam

## 🚀 How to Run

### Step 1: Install Dependencies

pip install tensorflow pandas matplotlib opencv-python

### Step 2: Train the Model

python train_model.py

This will:
- Load images and labels from the dataset
- Train a CNN
- Save the model as landmark_model.h5

⚠️ NOTE: `landmark_model.h5` is NOT included in this GitHub repo (it's >100MB).
You must generate it using `train_model.py`.

### Step 3: Predict on a Test Image

python predict.py

This will:
- Load test.jpg
- Predict facial landmark points using the trained model
- Show the image with red dots as landmarks

## 📦 Dataset

- landmarks_dataset/images/ → contains training images
- landmarks_dataset/labels.csv → contains (x, y) coordinates for each image

## ❌ Files Not Included in GitHub

File: `landmark_model.h5`  
Reason: GitHub’s upload limit is 100MB. You must generate this by training.

## 📸 Output

After prediction, the result is a grayscale image with red dots on facial keypoints.

## ✍️ Author

Sathvika Bogam  

## 📌 Notes

- Do NOT push `.h5` or `.keras` files to GitHub if over 100MB
- To submit via internship portal or Google Form, upload as a clean ZIP under 25MB
- This project is reproducible using the included dataset and scripts

## 📄 License

For academic and internship use only.
