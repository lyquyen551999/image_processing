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

st.set_page_config(page_title="🪞 Gương thần Pixel – Biểu cảm siêu hài", layout="wide")


st.title("🪞 Gương thần Pixel – Phân tích ảnh hài hước")

uploaded_file = st.file_uploader("📸 Vui lòng tải ảnh selfie của bạn:", type=["jpg", "jpeg", "png"])

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
        messages.append("😬 Hình như bạn đang... căng như dây đàn!")
    else:
        messages.append("😌 Bạn có vẻ đang thư giãn và chill~")
    if skin_tone == "pale":
        messages.append("😴 Da bạn nhợt nhạt ghê, thiếu ngủ không đó?")
    elif skin_tone == "red":
        messages.append("😡 Da đỏ rực vậy, mới đi cãi lộn ai à?")
    elif skin_tone == "normal":
        messages.append("😎 Màu da ổn áp, thần thái ổn định đó nhen!")
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
    st.success("✅ Đã chuyển đổi thành công sang Grayscale và HSV!")

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
        stress_level = "🧘‍♂️ Thư giãn như nước suối!"
    elif sharpness_score < 50:
        stress_level = "😐 Tạm ổn, nhưng hơi thiếu caffeine!"
    else:
        stress_level = "😵‍💫 Hình như bạn đang... căng như dây đàn!"
    st.markdown(f"**Stress Level Estimate:** `{stress_level}`")

    skin_tone = analyze_skin_tone(img_hsv)
    skin_message = {
        "red": "🔴 Có vẻ bạn đang nóng giận hoặc stress?",
        "pale": "🧊 Bạn hơi tái nhợt... thiếu vitamin D không nè?",
        "normal": "🌈 Da bạn khá cân bằng, chill nha!"
    }
    st.markdown(f"**Skin Tone Mood**: {skin_message[skin_tone]}")

    st.markdown("### 🔥 Burnout Simulator (Compressed Image)")
    compressed_image = simulate_burnout_effect(image)
    st.image(compressed_image, caption="📉 Mô phỏng trạng thái burnout", width=512)
    st.warning("🧠 Não bạn có vẻ đang 'nóng máy' đấy... nghỉ một tí đi nè!")

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
        "Poker face trình độ thượng thừa.",
        "Bạn đang suy ngẫm về vũ trụ và cuộc sống?",
        "Gương mặt này chuyên trị hội nghị và họp hành.",
        "Bạn có đang nghĩ về bữa tối không?",
        "Tâm hồn lạc vào cõi thiền?",
        "Vẻ mặt này là khi thấy deadline mà vẫn bình thản.",
        "Cân bằng hơn cả thiền sư.",
        "Mặt của người không quan tâm drama.",
        "Thần thái lạnh như băng.",
        "Giao diện đang cập nhật cảm xúc...",
        "Trí tuệ nhân tạo cũng không đoán được bạn nghĩ gì.",
        "Chắc bạn đang đứng hình 5 giây rồi."
    ],
    "happiness": [
        "Bạn đang tỏa sáng như idol K-pop!",
        "Cười tươi như mùa xuân về!",
        "Vừa được điểm cao hay được ăn món yêu thích vậy?",
        "Gương mặt của người vừa trúng vé số!",
        "Cười mà như quảng cáo kem đánh răng.",
        "Nụ cười này có thể chữa lành trái tim tan vỡ.",
        "Sáng bừng cả khung hình!",
        "Mặt vui như khi nghe tới giờ tan học.",
        "Chắc chắn có tin tốt vừa đến!",
        "Ai đó đang yêu đời dữ lắm!",
        "Bạn vừa làm ai đó vui chỉ với nụ cười.",
        "Cười mà không thấy răng là vui thật sự!"
    ],
    "surprise": [
        "Bạn vừa thấy điều gì đó cực sốc?",
        "Mắt chữ O, miệng chữ A chính hiệu!",
        "Ủa gì vậy trời???",
        "Giống như vừa phát hiện crush cũng thích mình!",
        "Biểu cảm khi mở phong bì lì xì thấy 500k.",
        "Có vẻ bạn vừa phát hiện ra bí mật vũ trụ!",
        "Mặt như vừa nhớ ra bỏ quên nồi cơm.",
        "Cảm xúc 'Ơ kìa' đang hiện hữu trên mặt bạn!",
        "Bạn vừa nghe tin: ngày mai được nghỉ học!",
        "Sự bất ngờ đang xâm chiếm linh hồn bạn.",
        "Bạn sắp hét lên đúng không?",
        "Cảm xúc không thể định nghĩa được!"
    ],
    "sadness": [
        "Ôi buồn như lá rơi mùa thu...",
        "Bạn có cần một cái ôm không?",
        "Tâm trạng như bị điểm thấp môn yêu thích?",
        "Gương mặt thất tình à?",
        "Bạn có đang nhớ mèo cũ không?",
        "Tối nay bạn cần 1 ly trà sữa an ủi.",
        "Nét buồn như ca sĩ hát ballad.",
        "Buồn nhưng vẫn đẹp nha!",
        "Chắc chắn có chuyện gì đó xảy ra rồi.",
        "Đôi mắt ngấn nước như sắp mưa.",
        "Gương mặt 'ngày chủ nhật tối'.",
        "Buồn vì hết phim hay để xem đúng không?"
    ],
    "anger": [
        "Bạn trông như đang bật mode 'đừng chọc tui'!",
        "Gương mặt của người vừa bị spoil phim!",
        "Biểu cảm của dân IT gặp bug 3 ngày chưa fix được.",
        "Bạn cần trà hay cần đá?",
        "Nét tức giận này đầy nội lực!",
        "Ai dám chọc bạn thì toang rồi.",
        "Chắc vừa đọc comment vô duyên?",
        "Sắp hóa Hulk đến nơi!",
        "Cáu nhưng vẫn đáng yêu nè!",
        "Gương mặt chờ cà phê nhưng người ta pha trà.",
        "Thở sâu... hít vào... thở ra...",
        "Đừng để bạn này bật chế độ chiến binh!"
    ],
    "disgust": [
        "Bạn vừa ngửi thấy mùi gì đó sai sai?",
        "Vẻ mặt khi ăn phải mít sầu riêng mà không thích.",
        "Biểu cảm 'ơ cái gì dạ???'",
        "Nét mặt như đang bị buộc nghe nhạc remix lúc sáng sớm.",
        "Bạn vừa nhìn thấy gì gây khó chịu vậy?",
        "Tôi thấy bạn vừa đánh giá 1 sao trong lòng.",
        "Gương mặt phản đối mọi thứ xung quanh.",
        "Bạn đang muốn rời khỏi nơi này gấp.",
        "Ánh nhìn ‘không thèm nói chuyện’.",
        "Có phải bạn vừa đọc status quá sến súa?",
        "Bạn vừa nghe câu chuyện toxic nào đó?",
        "Đây là phản ứng đúng khi gặp đồ ăn dở."
    ],
    "fear": [
        "Bạn vừa thấy hóa đơn tiền điện đúng không?",
        "Ánh mắt lo lắng của người đi thi không học bài.",
        "Gương mặt thấy deadline tới gần...",
        "Bạn trông như nghe tiếng mẹ gọi lúc chơi game.",
        "Bạn vừa thấy con gián bay đúng không!?",
        "Gương mặt 'chắc mình tắt bếp rồi đó ha...'",
        "Biểu cảm đầy lo âu, nhưng vẫn xinh!",
        "Đừng sợ, có tôi ở đây mà 😁",
        "Bạn vừa nhớ lại chuyện cũ đáng sợ à?",
        "Sự hoang mang hiện rõ trên gương mặt!",
        "Tâm trạng 'mình có nộp bài chưa nhỉ?'"
    ],
    "contempt": [
        "Cái nhìn khinh bỉ rất nhẹ nhưng đầy sát thương.",
        "Bạn đang khinh nhẹ ai đó đúng không?",
        "Mắt bạn vừa lật nhẹ một vòng tròn.",
        "Gương mặt ‘tôi hơn bạn ở cái thần thái’.",
        "Bạn như đang đóng vai phản diện lạnh lùng.",
        "Chắc bạn vừa nghe câu chuyện không đáng nghe.",
        "Ai đó vừa làm bạn mất kiên nhẫn?",
        "Cái nhìn của người từng trải và đã chán drama.",
        "Bạn cần cốc trà đá cho bớt khinh người ta kìa 😆",
        "Nét mặt này xứng đáng đứng đầu hội đồng!",
        "Bạn biết quá nhiều và không muốn nói ra.",
        "Đây là đỉnh cao của thái độ 'Ừ, tôi biết rồi!'"
    ]
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
    st.image(image, caption="📷 Ảnh gốc", width=512)

    faces = detect_faces_dnn(img_bgr)

    if not faces:
        st.warning("😶 Không tìm thấy khuôn mặt nào.")
    else:
        st.success(f"Đã phát hiện {len(faces)} khuôn mặt!")
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

            fun_comment = random.choice(fun_diagnosis_dict.get(emotion, ["Biểu cảm khó hiểu quá! 🤔"]))
            with col1 if idx % 2 == 0 else col2:
                st.image(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB),
                         caption=f"Mặt #{idx+1}: {emotion} {emoji_dict[emotion]} ({score:.2f})\n{fun_comment}",
                         width=512)

        st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB),
                 caption="📸 Gắn nhãn biểu cảm vui nhộn", width=512)
