import streamlit as st
from foot_estimate import drawCnt

def manual_a4_adjustment(a4_box, contours, contours_poly, img):
    st.subheader("Adjust A4 bounding box manually using sliders")
    max_width_a4 = img.shape[1]
    max_height_a4 = img.shape[0]

    ax = st.slider("A4 box X Position", 0, max_width_a4, a4_box[0])
    ay = st.slider("A4 box Y Position", 0, max_height_a4, a4_box[1])

    max_aw = max_width_a4 - ax
    max_ah = max_height_a4 - ay

    aw_default = min(a4_box[2], max_aw)
    ah_default = min(a4_box[3], max_ah)

    aw = st.slider("A4 box Width", 10, max_aw, aw_default)
    ah = st.slider("A4 box Height", 10, max_ah, ah_default)

    manual_a4_box = (ax, ay, aw, ah)
    st.success(f"Manual A4 bounding box set to (x={ax}, y={ay}, w={aw}, h={ah})")

    manual_pdraw = drawCnt(manual_a4_box, contours, contours_poly, img.copy())
    st.image(manual_pdraw, caption='A4 Paper Bounding Box (Manual Adjustment)', use_container_width=True)

    return manual_a4_box


def manual_foot_adjustment(fboundRect, fcnt, fcntpoly, fimg):
    st.subheader("Adjust Foot bounding box manually using sliders")
    max_width_foot = fimg.shape[1]
    max_height_foot = fimg.shape[0]

    fx = st.slider("Foot box X Position", 0, max_width_foot, fboundRect[0])
    fy = st.slider("Foot box Y Position", 0, max_height_foot, fboundRect[1])

    max_fw = max_width_foot - fx
    max_fh = max_height_foot - fy

    fw_default = min(fboundRect[2], max_fw)
    fh_default = min(fboundRect[3], max_fh)

    fw = st.slider("Foot box Width", 10, max_fw, fw_default)
    fh = st.slider("Foot box Height", 10, max_fh, fh_default)

    manual_foot_box = (fx, fy, fw, fh)
    st.success(f"Manual foot bounding box set to (x={fx}, y={fy}, w={fw}, h={fh})")

    manual_fdraw = drawCnt(manual_foot_box, fcnt, fcntpoly, fimg.copy())
    st.image(manual_fdraw, caption='Foot Edge Bounding Box (Manual Adjustment)', use_container_width=True)

    return manual_foot_box
