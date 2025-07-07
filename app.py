import streamlit as st
from deepface import DeepFace
import cv2
import av
import random
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

chat_responses = {
    "happy": ["You're glowing!", "Feeling great, huh? 😄"],
    "sad": ["I'm here for you 💙", "Want to talk about it?"],
    "angry": ["Deep breaths. Let's calm down.", "I'm listening 👂"],
    "neutral": ["How’s your day going?", "Let’s chat!"]
}

def get_chat_response(emotion, user_input):
    return random.choice(chat_responses.get(emotion, chat_responses["neutral"])) + f" You said: '{user_input}'"

class EmotionDetector(VideoTransformerBase):
    def __init__(self):
        self.last_emotion = "neutral"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        try:
            result = DeepFace.analyze(img, actions=["emotion"], enforce_detection=False)
            emotion = result[0]["dominant_emotion"]
            self.last_emotion = emotion
            cv2.putText(img, f"Emotion: {emotion}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except Exception as e:
            print("Emotion detection error:", e)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("Emotion Chatbot 🤖")

webrtc_ctx = webrtc_streamer(
    key="emotion",
    video_transformer_factory=EmotionDetector,
    async_transform=True
)

user_input = st.text_input("You:", key="input")
if user_input and webrtc_ctx.video_transformer:
    emotion = webrtc_ctx.video_transformer.last_emotion
    response = get_chat_response(emotion, user_input)
    st.write("Bot:", response)
