import os
import torch
import numpy as np
from PIL import Image
import streamlit as st
from drag_gan_edit_2 import drag_gan_edit
from streamlit_drawable_canvas import st_canvas
from StyleGAN2_Implementation import Generator, MappingNetwork

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
Z_DIM = 256  
W_DIM = 256
LOG_RESOLUTION = 7  

# --- Load models ---
gen  = Generator(LOG_RESOLUTION, W_DIM).to(DEVICE)
gen.load_state_dict(torch.load(os.path.join('stylegan2_checkpoints', 'gen.pth'), map_location=DEVICE))
mapping_network  = MappingNetwork(Z_DIM, W_DIM).to(DEVICE)
mapping_network.load_state_dict(torch.load(os.path.join('stylegan2_checkpoints', 'mapping_network.pth'), map_location=DEVICE))
gen.eval()
mapping_network.eval()

def generate_image(z, gen, mapping_network):
    with torch.no_grad():
        w = mapping_network(z)
        w_broadcast = [w for _ in range(LOG_RESOLUTION)]
        noise = []
        resolution = 4
        for i in range(LOG_RESOLUTION):
            n1 = torch.randn(1, 1, resolution, resolution).to(DEVICE) if i != 0 else None
            n2 = torch.randn(1, 1, resolution, resolution).to(DEVICE)
            noise.append((n1, n2))
            resolution *= 2
        img = gen(w_broadcast, noise)
        img = (img * 0.5 + 0.5).clamp(0, 1)
        img = img[0].permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        return img

st.title("DragGAN: Single Point Edit")

# --- Initialize session state ---
if "z" not in st.session_state:
    st.session_state.z = None
if "last_img" not in st.session_state:
    st.session_state.last_img = None
if "edit_done" not in st.session_state:
    st.session_state.edit_done = False

canvas_height = 512
canvas_width = 512

if st.button("Generate New Image"):
    st.session_state.z = torch.randn(1, Z_DIM).to(DEVICE)
    st.session_state.last_img = generate_image(st.session_state.z, gen, mapping_network)
    st.session_state.edit_done = False

if st.session_state.last_img is not None and not st.session_state.edit_done:
    img = st.session_state.last_img
    img_h, img_w = img.shape[:2]
    img_for_canvas = Image.fromarray(img).resize((canvas_width, canvas_height))
    st.image(img, caption="Original Image", use_container_width=True)

    st.write("Draw a source point and a target point (both red):")

    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=6,
        stroke_color="#e00",
        background_image=img_for_canvas,
        update_streamlit=True,
        height=canvas_height,
        width=canvas_width,
        drawing_mode="circle",
        key="canvas",
    )

    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        if len(objects) == 2:
            scale_x = img_w / canvas_width
            scale_y = img_h / canvas_height
            source = (int(objects[0]["left"] * scale_x), int(objects[0]["top"] * scale_y))
            target = (int(objects[1]["left"] * scale_x), int(objects[1]["top"] * scale_y))
            st.write(f"Source: {source}, Target: {target}")

            steps = st.slider("Number of optimization steps", min_value=100, max_value=3000, step=100, value=1000)
            log_lr = st.slider("Learning Rate (10^x)", min_value=-5.0, max_value=-1.0, step=0.1, value=-3.0, format="%.1f")
            lr = 10 ** log_lr
            st.write(f"Learning Rate: {lr:.5e}")

            # New tracking-specific parameter
            threshold = st.slider("Stopping threshold (px)", min_value=1.0, max_value=10.0, value=2.0, step=0.5)



            if st.button("Run DragGAN Edit"):
                with st.spinner("Optimizing with tracked point movement..."):
                    img_tensor, w = drag_gan_edit(
                        gen, mapping_network, st.session_state.z, [source], [target],
                        num_steps=steps,
                        lr=lr,
                        device=DEVICE,
                        is_w_input=False,
                        log_resolution=LOG_RESOLUTION,
                        stopping_threshold=threshold
                    )
                last_img = img_tensor.permute(1, 2, 0).numpy()
                last_img = (last_img * 255).astype(np.uint8)
                st.session_state.last_img = last_img
                st.session_state.edit_done = True
                st.success("Editing complete!")


if st.session_state.edit_done:
    st.image(st.session_state.last_img, caption="Final Edited Image", use_container_width=True)

if st.button("Reset All"):
    st.session_state.z = None
    st.session_state.last_img = None
    st.session_state.edit_done = False
    st.rerun()
