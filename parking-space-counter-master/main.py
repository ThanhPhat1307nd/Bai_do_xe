import cv2
import numpy as np
import streamlit as st
from util import get_parking_spots_bboxes, empty_or_not

# ==== Cấu hình Streamlit ====
st.set_page_config(page_title="Parking Detection", layout="wide")
st.title("🚗 Parking Spot Detection Demo")

# ==== Đường dẫn video & mask ====
mask_path = 'mask_1920_1080.png'
video_path = 'https://github.com/ThanhPhat1307nd/parking-video-storage/releases/download/v1/parking_1920_1080_loop.mp4'

# ==== Đọc dữ liệu ====
mask = cv2.imread(mask_path, 0)
cap = cv2.VideoCapture(video_path)

connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
spots = get_parking_spots_bboxes(connected_components)

spots_status = [None for _ in spots]
diffs = [None for _ in spots]

previous_frame = None
frame_nmr = 0
step = 15
ret = True

# ==== Placeholder hiển thị video và thông tin ====
info_placeholder = st.empty()   # nơi hiển thị số chỗ trống
frame_placeholder = st.empty()  # nơi hiển thị video

# ==== Nút dừng ====
stop = st.button("⏹ Stop Processing")

# ==== Xử lý video ====
while ret and not stop:
    ret, frame = cap.read()
    if not ret:
        st.warning("Video has ended or cannot be read.")
        break

    if frame_nmr % step == 0 and previous_frame is not None:
        for spot_idx, spot in enumerate(spots):
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            diffs[spot_idx] = np.abs(np.mean(spot_crop) - np.mean(previous_frame[y1:y1 + h, x1:x1 + w, :]))

    if frame_nmr % step == 0:
        if previous_frame is None:
            arr_ = range(len(spots))
        else:
            arr_ = [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > 0.4]

        for spot_idx in arr_:
            x1, y1, w, h = spots[spot_idx]
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            spot_status = empty_or_not(spot_crop)
            spots_status[spot_idx] = spot_status

    if frame_nmr % step == 0:
        previous_frame = frame.copy()

    # ==== Vẽ khung chỗ đỗ xe ====
    for spot_idx, (x1, y1, w, h) in enumerate(spots):
        color = (0, 255, 0) if spots_status[spot_idx] else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)

    # ==== Thông tin số chỗ trống ====
    empty_count = sum(spots_status)
    total_spots = len(spots_status)

    cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
    cv2.putText(frame, f'Empty: {empty_count} / {total_spots}',
                (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # ==== Hiển thị lên Streamlit ====
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

    # 🔹 Hiển thị số chỗ trống ra ngoài (trên web, cập nhật realtime)
    info_placeholder.markdown(
        f"### 🅿️ Vị trí trống hiện tại: **{empty_count} / {total_spots}**",
        unsafe_allow_html=True
    )

    frame_nmr += 1

cap.release()
st.success("✅ Video processing completed.")
