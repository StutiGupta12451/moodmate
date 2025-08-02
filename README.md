# 🧠 Mood Mate – Emotion-Based Activity & Music Recommender

**Mood Mate** is an interactive, AI-powered web application that detects your emotional state through a webcam snapshot or a short questionnaire and recommends personalized activities or music based on your mood.

---

## 🌟 Key Features

### 🎥 Emotion Detection via Webcam

* Captures your image using your device's webcam.
* Uses a pre-trained deep learning model (`model.h5`) to detect facial expressions.
* Identifies one of seven emotions: **Angry**, **Disgust**, **Fear**, **Happy**, **Sad**, **Surprise**, or **Neutral**.

### 🧠 Intelligent Recommendations

* Once your mood is detected, it uses **LLM-based prompting** (via **LLaMA3**) to generate personalized recommendations.
* Suggestions include **fun activities**, **calming techniques**, **productivity tips**, or **emotion-regulating tasks**.

### 📋 Optional Mood Questionnaire

* Don’t want to use the webcam? Use the built-in questionnaire to express your feelings manually.
* Input includes:

  * Mood rating (1–10)
  * Mood description
  * Frequency and causes of feelings
  * Stress response style
* The app interprets the responses using LLaMA3 and provides relevant support tips.

---

## 🚀 How It Works

1. **User Input**

   * Through webcam image capture or questionnaire.
2. **Emotion Detection**

   * Webcam image is processed using OpenCV.
   * Face is detected using Haar Cascades.
   * A deep learning model (`model.h5`) classifies the emotion.
3. **Prompt Generation**

   * Based on detected emotion or form inputs, a tailored natural language prompt is generated.
4. **LLM API Call**

   * Prompt is sent to a **local LLaMA3 API** (`localhost:11434`) for generating recommendations.
5. **Output Display**

   * The app shows personalized advice or activity suggestions based on mood.

---

## 🛠️ Tech Stack

* **Frontend/UI**: [Streamlit](https://streamlit.io/)
* **Model Serving**: [Keras](https://keras.io/) with `model.h5` for facial emotion classification
* **Computer Vision**: [OpenCV](https://opencv.org/) for image capture and face detection
* **Large Language Model (LLM)**: [LLaMA3](https://llama.meta.com/) via local server for text generation
* **Backend Communication**: Python `requests` to interact with the LLM API

---

## 📂 Project Structure (Overview)

```
mood_mate/
│
├── app.py               # Main Streamlit application file
├── model.h5             # Trained Keras model for emotion detection
└── README.md            # Project documentation (you're reading it!)
```

---

## 🧪 Emotions Detected

| Emotion  | Description                      |
| -------- | -------------------------------- |
| Happy    | Positive and energetic mood      |
| Sad      | Low energy or downcast feelings  |
| Angry    | Frustration or agitation         |
| Disgust  | Aversion or repulsion            |
| Fear     | Anxious or worried state         |
| Surprise | Unexpected emotion – good or bad |
| Neutral  | Calm or indifferent              |

---

## 📌 Use Cases

* Emotional self-check-ins for students, professionals, or general users.
* Mood-based productivity boosters or calming exercises.
* Mental health support and mood journaling.

---

## 🔧 Requirements

Ensure the following are installed:

```bash
pip install streamlit keras opencv-python numpy requests
```

Also, you must be running a **LLaMA3 server locally** at `http://localhost:11434/api/generate` with support for the `llama3` model.

---

## 📸 Notes

* Webcam access is required for real-time image-based mood detection.
* The emotion model expects grayscale 48x48 facial images.


## 📜 License

This project is open-source and free to use under the MIT License.

