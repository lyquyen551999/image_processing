import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
import urllib.request
import random

st.set_page_config(page_title="Pixel Mirror – Hilariously Expressive Faces", layout="wide")


st.title("Analyze Your Selfie with Humor")

uploaded_file = st.file_uploader("📸 Upload your selfie to begin the fun!", type=["jpg", "jpeg", "png"])

def analyze_skin_tone(hsv_img):
    h, s, v = cv2.split(hsv_img)
    h_mean = np.mean(h)
    s_mean = np.mean(s)
    v_mean = np.mean(v)
    if h_mean < 20 and s_mean > 100:
        return "red"
    elif h_mean > 90 and v_mean > 150:
        return "pale"
    elif s_mean < 60:
        return "pale"
    else:
        return "normal"

def simulate_burnout_effect(image_pil):
    buffer = io.BytesIO()
    image_pil.save(buffer, format="JPEG", quality=5)
    buffer.seek(0)
    compressed_img = Image.open(buffer)
    return compressed_img

def generate_diagnosis(sharpness_score, skin_tone):
    messages = []
    if sharpness_score > 5.0:
        messages.append("😬 You seem... tense like a guitar string!")
    else:
        messages.append("😌 ou seem totally relaxed and at ease~")
    if skin_tone == "pale":
        messages.append("😴 You look really pale... haven’t been sleeping, huh?")
    elif skin_tone == "red":
        messages.append("😡 Your face is super red—just had an argument or what?")
    elif skin_tone == "normal":
        messages.append("😎 Skin tone looks great, vibe’s steady and strong!")
    final_message = "\n".join(messages)
    return final_message

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    st.image(image, caption="📷 Ảnh gốc",width=512)
    st.image(img_gray, caption="🔍 Ảnh Grayscale", channels="GRAY", width=512)
    st.image(cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB), caption="🎨 Ảnh HSV", width=512)
    st.success("✅ Successfully converted to Grayscale and HSV!")

    blurred = cv2.GaussianBlur(img_gray, (9, 9), 0)
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
    sobel_combined = cv2.magnitude(sobelx, sobely)
    sobel_combined = np.uint8(np.clip(sobel_combined, 0, 255))

    st.subheader("🔬 Spatial Domain Filtering")
    st.image(blurred, caption="✨ Gaussian Blurred Image", channels="GRAY", width=512)
    st.image(sobel_combined, caption="🧠 Sobel Edge Detection", channels="GRAY", width=512)

    dft = cv2.dft(np.float32(img_gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    spectrum = 20 * np.log(magnitude + 1)
    spectrum_norm = cv2.normalize(spectrum, None, 0, 255, cv2.NORM_MINMAX)
    spectrum_uint8 = np.uint8(spectrum_norm)
    sharpness_score = np.mean(spectrum)

    st.subheader("⚡ Frequency Domain Analysis")
    st.image(spectrum_uint8, caption="📈 Frequency Spectrum (DFT)", width=512)

    if sharpness_score < 20:
        stress_level = "🧘‍♂️ Relaxed like a mountain stream!"
    elif sharpness_score < 50:
        stress_level = "😐 Not bad, but looks like you're low on caffeine!"
    else:
        stress_level = "😵‍💫 You seem... tense like a guitar string!"
    st.markdown(f"**Stress Level Estimate:** `{stress_level}`")

    skin_tone = analyze_skin_tone(img_hsv)
    skin_message = {
        "red": "🔴  You look angry or maybe just super stressed?",
        "pale": "🧊 You look a bit pale... missing some vitamin D?",
        "normal": "🌈 Your skin looks balanced—chill vibes detected!"
    }
    st.markdown(f"**Skin Tone Mood**: {skin_message[skin_tone]}")

    st.markdown("### 🔥 Burnout Simulator (Compressed Image)")
    compressed_image = simulate_burnout_effect(image)
    st.image(compressed_image, caption="📉 Burnout Simulation", width=512)
    st.warning("🧠 Your brain seems overheated... take a quick break!")

    diagnosis = generate_diagnosis(sharpness_score, skin_tone)
    st.markdown("### 🧠 Pixel Mirror Diagnosis")
    st.success(diagnosis)

# ========== TÍCH HỢP NHẬN DIỆN BIỂU CẢM & CHẨN ĐOÁN HÀI HƯỚC ========== #




# Tải model nếu chưa có
if not os.path.exists("emotion-ferplus-8.onnx"):
    urllib.request.urlretrieve("https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx", "emotion-ferplus-8.onnx")
if not os.path.exists("deploy.prototxt"):
    urllib.request.urlretrieve("https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt", "deploy.prototxt")
if not os.path.exists("res10_300x300_ssd_iter_140000_fp16.caffemodel"):
    urllib.request.urlretrieve("https://github.com/spmallick/learnopencv/raw/master/FaceDetectionComparison/models/res10_300x300_ssd_iter_140000_fp16.caffemodel", "res10_300x300_ssd_iter_140000_fp16.caffemodel")

emotion_labels = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]
emoji_dict = {
    "neutral": "😐", "happiness": "😄", "surprise": "😲", "sadness": "😢",
    "anger": "😠", "disgust": "🤢", "fear": "😱", "contempt": "🙄"
}

# Hơn 100 câu chẩn đoán, ngẫu nhiên
fun_diagnosis_dict = {
    "neutral": [
        "A master-level poker face.",
        "Are you contemplating the universe and life?",
        "This face belongs in boardrooms and meetings.",
        "Are you thinking about dinner right now?",
        "Your soul has wandered into a meditative state?",
        "That’s the look of seeing a deadline and remaining calm.",
        "More balanced than a Zen master.",
        "A face immune to all drama.",
        "Cold vibes. Ice cold.",
        "Emotion system is updating...",
        "Even AI can’t figure out what you’re thinking.",
        "You might be buffering for five seconds straight.",
    ],
    "happiness": [
        "You're glowing like a K-pop idol!",
        "Smiling like spring just arrived!",
        "Did you just ace a test or eat your favorite food?",
        "That’s the face of a lottery winner!",
        "A toothpaste commercial smile.",
        "That smile could heal a broken heart.",
        "Lighting up the whole frame!",
        "The joy of hearing the school bell ring!",
        "You clearly just got some great news!",
        "Someone's loving life a lot right now!",
        "Your smile just made someone’s day.",
        "Smiling so big your teeth disappeared!",
    ],
    "surprise": [
        "Did you just witness something shocking?",
        "Classic wide eyes and dropped jaw!",
        "Wait, WHAT???",
        "Looks like you just found out your crush likes you back!",
        "The face when your red envelope has 500K inside.",
        "Did you just unlock a secret of the universe?",
        "That moment you remember you left the rice cooker on.",
        "'Uh-oh' is written all over your face!",
        "You just heard tomorrow’s a day off, didn’t you?",
        "Surprise is taking over your soul.",
        "You're about to scream, right?",
        "That expression can’t even be defined!",
    ],
    "sadness": [
        "As sad as leaves falling in autumn...",
        "Do you need a hug?",
        "Like you got a low grade in your favorite subject.",
        "Heartbroken face detected.",
        "Missing your old cat, perhaps?",
        "Time for a comforting milk tea tonight.",
        "That ballad-singer sorrow vibe.",
        "Sad but still cute!",
        "Something definitely happened, huh?",
        "Eyes filled with almost-tears.",
        "Sunday night face mode activated.",
        "Out of good shows to binge, aren’t you?",
    ],
    "anger": [
        "You look like you're in 'don’t mess with me' mode!",
        "The face of someone who just got a movie spoiler!",
        "Classic IT face after 3 days debugging.",
        "Do you need tea or ice?",
        "This anger has some serious energy.",
        "Whoever annoyed you is in big trouble.",
        "Did you just read a really dumb comment?",
        "Hulk mode incoming!",
        "Furious but still adorable!",
        "Wanted coffee but got tea instead.",
        "Deep breaths… in… and out…",
        "Don’t trigger warrior mode!",
    ],
    "disgust": [
        "Did something just smell... off?",
        "That’s the face of eating durian against your will.",
        "‘What the heck?’ face engaged.",
        "Like being forced to listen to EDM at 6 AM.",
        "What did you just witness that was so cringe?",
        "That’s a mental 1-star rating right there.",
        "Rejecting everything around you.",
        "You're ready to leave. Immediately.",
        "Eyes saying 'not worth my words.'",
        "Did you just read a super cheesy status?",
        "Did someone just tell you a toxic story?",
        "This is the right reaction to bad food.",
    ],
    "fear": [
        "Was that your electricity bill just now?",
        "Exam face: didn’t study but showed up anyway.",
        "Deadline is creeping up, isn’t it?",
        "That moment when you hear Mom call during gaming.",
        "Wait—was that a flying cockroach!?",
        "Pretty sure you left the stove on, huh?",
        "Worried look, but still cute though!",
        "Don’t worry, I’m here with you 😁",
        "Just remembered a scary old memory?",
        "Panic is written all over your face!",
        "'Did I submit the assignment?' vibes.",
    ],
    "contempt": [
        "A subtle yet deadly judging stare.",
        "Are you silently judging someone?",
        "Your eyes just did a full 360 roll.",
        "Face says, ‘I’m better, obviously.’",
        "Looking like a cold-hearted movie villain.",
        "Did someone say something not worth your time?",
        "Someone clearly tested your patience.",
        "That experienced look that’s over all the drama.",
        "You need iced tea to cool that judgment 😆",
        "This expression belongs on a council throne.",
        "You know too much to even speak.",
        "Ultimate expression of ‘Yeah, I know already.’",
    ],
}

def detect_faces_dnn(image_bgr):
    net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")
    h, w = image_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(image_bgr, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            faces.append((x1, y1, x2 - x1, y2 - y1))
    return faces

def preprocess_face(face_img):
    face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (64, 64))
    face = face.astype(np.float32)
    face = face[np.newaxis, np.newaxis, :, :]
    return face

def predict_emotion(face_blob):
    net = cv2.dnn.readNetFromONNX("emotion-ferplus-8.onnx")
    net.setInput(face_blob)
    output = net.forward()
    emotion_idx = np.argmax(output)
    return emotion_labels[emotion_idx], float(np.max(output))

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    st.image(image, caption="📷 Raw Input Image", width=512)

    faces = detect_faces_dnn(img_bgr)

    if not faces:
        st.warning("😶 No faces detected.")
    else:
        st.success(f"Detected {len(faces)} face(s)!")
        col1, col2 = st.columns(2)
        annotated_img = img_bgr.copy()

        for idx, (x, y, w, h) in enumerate(faces):
            face_crop = img_bgr[y:y+h, x:x+w]
            if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
                continue

            blob = preprocess_face(face_crop)
            emotion, score = predict_emotion(blob)

            cv2.rectangle(annotated_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(annotated_img, f"{emotion}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            fun_comment = random.choice(fun_diagnosis_dict.get(emotion, ["Expression too mysterious to decode! 🤔"]))
            with col1 if idx % 2 == 0 else col2:
                st.image(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB),
                         caption=f"Face #{idx+1}: {emotion} {emoji_dict[emotion]} ({score:.2f})\\n{fun_comment}",
                         width=512)

        st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB),
                 caption="📸 Fun Emotion Labels", width=512)
