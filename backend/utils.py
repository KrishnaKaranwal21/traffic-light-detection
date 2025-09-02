import os

def get_demo_video():
    # Path relative to your repo
    demo_path = os.path.join("backend", "assets", "demo.mp4")

    if not os.path.exists(demo_path):
        raise FileNotFoundError(f"Demo video not found at {demo_path}")

    return demo_path
