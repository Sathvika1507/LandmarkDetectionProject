# 🗺️ Landmark Detection using TensorFlow

This internship project demonstrates facial landmark detection using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The model is trained to predict key facial points from grayscale images.

---

## 📁 Project Structure

LandmarkDetectionProject/
├── train_model.py # Trains the CNN and saves the model
├── predict.py # Loads the model and predicts keypoints
├── test.jpg # Sample image to test prediction
├── landmarks_dataset/
│ ├── images/
│ │ └── sample.jpg # Sample training image
│ └── labels.csv # CSV file with landmark coordinates
└── README.md # You're reading it!


---

## 🧠 Model Details

- **Input**: 96x96 grayscale image  
- **Architecture**: Conv2D → MaxPooling → Flatten → Dense  
- **Output**: 30 facial landmark coordinates (15 x, y pairs)  
- **Loss**: Mean Squared Error (MSE)  
- **Optimizer**: Adam

---

## 🚀 How to Run

### Step 1: Install Dependencies

```bash
pip install tensorflow pandas matplotlib opencv-python
Step 2: Train the Model
python train_model.py
This will:

Read the dataset from landmarks_dataset/
Train the CNN model
Save it as landmark_model.h5
⚠️ Note: landmark_model.h5 is not included in this repository due to GitHub's 100MB file size limit. You must generate it by running train_model.py.
Step 3: Predict Landmarks
python predict.py
This will:

Load the trained model
Read test.jpg
Predict landmark positions
Plot and display them on the image
📦 Dataset

landmarks_dataset/images/: Contains training images
landmarks_dataset/labels.csv: Contains corresponding landmark (x, y) coordinates
❌ Files Not Included

File	Reason
landmark_model.h5	Exceeds GitHub 100MB limit. Re-create it using train_model.py
📸 Sample Output

Once trained and predicted, the program overlays red dots on the image to indicate detected facial landmarks.

✍️ Author

Sathvika Bogam
ECE Intern Project – IIIT Nagpur

📌 Notes

Do not upload .h5 or .keras files to GitHub if they exceed 100MB.
This project is also ready as a clean ZIP (<25MB) for internship submission via Google Forms.
To regenerate the model: python train_model.py
📄 License

For academic and internship use only.


---

### ✅ Now You’re Done

If you want:
- 📄 Report (PDF or Word)
- 📊 Internship PPT
- 📦 Final ZIP (under 25MB)

Just say:  
**“Yes, send final report + ppt + zip”**  
And I’ll send all ready-to-submit materials!
