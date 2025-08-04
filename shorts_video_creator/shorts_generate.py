# generate_shorts.py

import os
import shutil
import cv2
from PIL import Image, ImageDraw, ImageFont
import pillow_heif
pillow_heif.register_heif_opener()
import subprocess
import tempfile
import exifread
from geopy.geocoders import Nominatim
from geopy.location import Location
from textwrap import wrap
import zipfile
import uuid

import gradio as gr

geolocator = Nominatim(user_agent="shorts-caption-generator")
VIDEO_SIZE = (1080, 1920)

# --- Utilities ---
def format_location(location: Location):
    parts = location.raw.get('address', {})
    city = parts.get('city') or parts.get('town') or parts.get('village') or ''
    state = parts.get('state') or ''
    return ", ".join([c for c in [city, state] if c])

def extract_location_from_image(image_path):
    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f, stop_tag="GPS GPSLongitude")

        if "GPS GPSLatitude" in tags and "GPS GPSLongitude" in tags:
            def convert(value):
                d, m, s = [float(v.num) / float(v.den) for v in value.values]
                return d + (m / 60.0) + (s / 3600.0)

            lat = convert(tags["GPS GPSLatitude"])
            lon = convert(tags["GPS GPSLongitude"])
            if tags.get("GPS GPSLatitudeRef").values != "N":
                lat = -lat
            if tags.get("GPS GPSLongitudeRef").values != "E":
                lon = -lon

            location = geolocator.reverse((lat, lon), exactly_one=True, language='en')
            if location:
                return format_location(location)
    except Exception as e:
        print(f"[⚠️ GPS error] {image_path}: {e}")
    return None

def resize_and_pad_pil(image: Image.Image) -> Image.Image:
    img_ratio = image.width / image.height
    target_ratio = VIDEO_SIZE[0] / VIDEO_SIZE[1]
    new_width, new_height = VIDEO_SIZE

    if img_ratio > target_ratio:
        new_width = VIDEO_SIZE[0]
        new_height = round(new_width / img_ratio)
    else:
        new_height = VIDEO_SIZE[1]
        new_width = round(new_height * img_ratio)

    resized = image.resize((new_width, new_height), Image.LANCZOS)
    final_img = Image.new("RGB", VIDEO_SIZE, (0, 0, 0))
    final_img.paste(resized, ((VIDEO_SIZE[0] - new_width) // 2, (VIDEO_SIZE[1] - new_height) // 2))
    return final_img

def add_caption(image_path, output_path, enable_caption):
    img = Image.open(image_path).convert("RGB")
    img = resize_and_pad_pil(img)
    
    if not enable_caption:
        img.save(output_path)
        return

    caption = extract_location_from_image(image_path)
    if not caption:
        img.save(output_path)
        return

    draw = ImageDraw.Draw(img)
    try:
        font_path = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        font = ImageFont.truetype(font_path, 40)
    except:
        font = ImageFont.load_default()

    lines = wrap(caption, width=40)
    y = int(VIDEO_SIZE[1] * 0.7)
    line_height = 50

    for i, line in enumerate(lines):
        w = draw.textlength(line, font=font)
        x = (VIDEO_SIZE[0] - w) // 2
        y_offset = y + i * line_height
        draw.text((x, y_offset), line, font=font, fill="white", stroke_fill="black", stroke_width=2)

    img.save(output_path)

def resize_frame(frame):
    h, w = frame.shape[:2]
    target_w, target_h = VIDEO_SIZE
    frame_ratio = w / h
    target_ratio = target_w / target_h

    if frame_ratio > target_ratio:
        new_w = target_w
        new_h = int(new_w / frame_ratio)
    else:
        new_h = target_h
        new_w = int(new_h * frame_ratio)

    resized = cv2.resize(frame, (new_w, new_h))
    return cv2.copyMakeBorder(resized, (target_h-new_h)//2, (target_h-new_h+1)//2,
                              (target_w-new_w)//2, (target_w-new_w+1)//2,
                              borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])

def process_inputs(files, caption_on, fps, music_file):
    temp_dir = tempfile.mkdtemp()
    frame_dir = os.path.join(temp_dir, "frames")
    os.makedirs(frame_dir, exist_ok=True)
    frame_paths = []

    for file in files:
        name = file.name.lower()
        ext = os.path.splitext(name)[-1].lower()

        if ext == ".zip":
            with zipfile.ZipFile(file.name, 'r') as zf:
                zf.extractall(temp_dir)
            extracted_files = []
            for root, _, fnames in os.walk(temp_dir):
                for f in fnames:
                    extracted_files.append(os.path.join(root, f))
            files.extend(open(p, "rb") for p in extracted_files if os.path.isfile(p))
            continue

        elif ext in (".jpg", ".jpeg", ".png", ".heic", ".heif"):
            uid = uuid.uuid4().hex
            out_path = os.path.join(frame_dir, f"{uid}.jpg")
            add_caption(file.name, out_path, caption_on)
            frame_paths.append(out_path)

        elif ext in (".mp4", ".mov"):
            cap = cv2.VideoCapture(file.name)
            file_fps = cap.get(cv2.CAP_PROP_FPS)
            interval = max(1, int(file_fps / fps))  # frame step
            count = 0
            idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if count % interval == 0:
                    resized = resize_frame(frame)
                    out_path = os.path.join(frame_dir, f"{os.path.basename(file.name)}_{idx}.jpg")
                    cv2.imwrite(out_path, resized)
                    frame_paths.append(out_path)
                    idx += 1
                count += 1
            cap.release()

    if not frame_paths:
        shutil.rmtree(temp_dir)
        raise ValueError("No valid media to process")

    final_video = os.path.join(temp_dir, "shorts_final.mp4")
    out = cv2.VideoWriter(final_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, VIDEO_SIZE)
    for fp in sorted(frame_paths):
        out.write(cv2.imread(fp))
    out.release()

    if music_file:
        final_with_music = os.path.join(temp_dir, "shorts_with_music.mp4")
        subprocess.run([
            "ffmpeg", "-y",
            "-i", final_video,
            "-i", music_file.name,
            "-shortest",
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            final_with_music
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return final_with_music
    else:
        return final_video

# --- Gradio UI ---
def generate_shorts(files, caption, fps, music):
    try:
        output_path = process_inputs(files, caption, fps, music)
        return (
            "✅ Your video is ready. Click below to download.",
            output_path
        )
    except Exception as e:
        return (f"❌ Error: {e}", None)


with gr.Blocks() as demo:
    gr.Markdown("## 🎬 Shorts Video Creator")
    gr.Markdown("Upload photos/videos or a zip file of assets to generate 1080x1920 Shorts with optional GPS-based location captions and music.")

    with gr.Row():
        files = gr.File(
            label="Upload images/videos or a ZIP",
            file_types=[".jpg", ".jpeg", ".png", ".heic", ".heif", ".mp4", ".mov", ".zip",
                        ".JPG", ".JPEG", ".PNG", ".HEIC", ".HEIF", ".MP4", ".MOV", ".ZIP"],
            file_count="multiple")
        music = gr.File(label="(Optional) Background Music", file_types=[".mp3"], file_count="single")

    with gr.Row():
        caption = gr.Checkbox(label="Smart GPS-based captions", value=True)
        fps = gr.Slider(label="FPS (lower = longer per frame)", minimum=0.2, maximum=5.0, value=1.0, step=0.1)

    generate = gr.Button("▶️ Generate Shorts")

    status = gr.Markdown()
    download = gr.File(label="📥 Download Your Shorts Video")

    generate.click(
        fn=generate_shorts,
        inputs=[files, caption, fps, music],
        outputs=[status, download]
    )

demo.launch()
