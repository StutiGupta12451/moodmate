import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import requests


st.set_page_config(page_title="Mood Mate", layout="centered")

st.markdown(
    """
    <h1 style='
        text-align: center;
        color: #ffffff;
        font-family: "Poppins", sans-serif;
        font-size: 3.5rem;
        background-image: linear-gradient(to right, #ffffff 0%, #888888 100%);
        color: transparent;
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    '>
        Mood Mate
    </h1>
    """,
    unsafe_allow_html=True
)
def recommender(prompt):
    with st.spinner("Let me work on your input"):
        response=requests.post("http://localhost:11434/api/generate",
                                    json={
                                        "model":"llama3",
                                        "prompt":prompt,
                                        "stream": False
                                        }
                                        )
        result=response.json()
        st.write(result['response'])
tab1,tab2,tab3=st.tabs(['Overview','Emotion Detection',"Questionnaire"])
with tab1:
    st.subheader("Welcome to Mood Mate")
    st.write("""
                    Mood Mate – Emotion-Based Activity & Music Recommender
    Mood Mate is a smart, interactive web application that detects your emotional state through a single image capture and offers personalized task suggestions and Spotify playlist recommendations to match your mood.

    🌟 Key Features:
    🎥 One-Click Emotion Detection:
    Using a trained deep learning model (model.h5) and your webcam, the app captures a single image and accurately identifies your emotion (e.g., Happy, Sad, Angry, etc.).

    💡 Smart Activity Recommendations:
    Once your emotion is detected, Mood Mate uses LLM-based prompting (via LLaMA3) to suggest custom tasks suited to your current emotional state — whether you're looking to calm down, stay productive, or lift your spirits.

    🎯 Use Case:
    Perfect for students, professionals, or anyone looking for a quick emotional check-in and personalized support — whether through calming tasks or fun activities.
                    """)
with  tab2:
    st.subheader("Try to click a pick of you!")
    
        
    


    def emotiondetect():
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            st.warning("Failed to capture image.")
            return None

        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            st.warning("No face detected.")
            return None

        
        (x, y, w, h) = faces[0]
        face = gray[y:y+h, x:x+w]

        
        face = cv2.resize(face, (48, 48))  
        face = face.astype("float") / 255.0
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)  

        if model.input_shape[-1] == 3:
            face = np.repeat(face, 3, axis=-1)  

        pred = model.predict(face)
        emotion = emotion_labels[np.argmax(pred)]
        confidence = np.max(pred)
        return emotion,confidence
        

    model = load_model("model.h5")

    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    if st.button("Capture image📷"):
        result = emotiondetect()
        if result is not None:
            emotion_acc,confident=result
            confident=confident*100
            st.success(f"Hmm I seem {round(confident)}% sure that your mood is {emotion_acc}")
            prompt=""
            if emotion_acc=='Happy':
                prompt="""I am feeling happy and upbeat! Recommend 5
                fun or productive things I can do to make the most of this mood."""
                recommender(prompt)
                
            elif emotion_acc=='Sad':
                prompt="""I’m feeling sad. Suggest 5 gentle, 
                comforting tasks or uplifting activities I can do to feel a bit better."""
                recommender(prompt)
                
                
            elif emotion_acc=='Angry':
                prompt="""I’m feeling angry right now. Suggest 5 healthy
                ways to release this anger or shift
                my focus. Maybe some physical activities, creative outlets,
                or calming techniques."""
                recommender(prompt)
                
                
            elif emotion_acc=='Disgust':
                prompt="""I’m feeling disgusted or uncomfortable. Suggest 5 ways to reset my
                mood or distract myself with something refreshing and uplifting."""
                recommender(prompt)
                
                
            elif emotion_acc=='Fear':
                prompt="""I’m feeling anxious or afraid. Recommend 5 tasks that can calm 
                me down or make me feel safe and in control."""
                recommender(prompt)
                
                
            elif emotion_acc=='Surprise':
                prompt="""I’m feeling surprised or caught off guard.
                Suggest 5 tasks that help me process this unexpected feeling—positively 
                or calmly."""
                recommender(prompt)
                
                
            elif emotion_acc=='Neutral':
                prompt="""I feel neutral or in-between emotions. Suggest 5 tasks
                that are mildly engaging or help me discover what I actually want to do."""
                recommender(prompt)
                
                
            else:
                st.warning("Oh Oh some problem occurred!")  
        else:
            st.warning("Oh Oh some problem occurred!")
with tab3:
    st.header("Questionnaire")
    rate=st.number_input("Rate your mood from 1-10")
    mood=st.text_input("Describe your mood in one word (e.g., happy, sad, anxious)")
    feel=st.text_input("How often do you feel this way? (e.g., rarely, sometimes, often)")
    current=st.text_input("How do you feel about your current situation? (e.g., positive, negative, neutral)")
    words=st.text_input("What words best describe your feelings towards others? (e.g., supportive, critical, indifferent)")
    events=st.text_input("What recent events have influenced your mood? (open-ended)")
    stress=st.text_input("How do you typically respond to stress? (e.g., withdrawal, seeking support")
    prompt1=f"""i'd rate my mood {rate}, i'd describe my mood 
        as {mood},i feel {feel},i feel {current} about my current situation,
        i would describe my feelings towards others as {words},{events} these are the events
        which influenced  my  mood,i typically respond to my stress by {stress}
        can you help me out with these problems and suggest me some ways to deal with it
        """ 
    if st.button("Submit"):
        recommender(prompt1)
        
        