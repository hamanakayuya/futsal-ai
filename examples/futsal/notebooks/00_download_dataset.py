from dotenv import load_dotenv
import os
from roboflow import Roboflow

def main():
    # notebooks/.env を読む（カレントディレクトリに .env があればOK）
    load_dotenv()

    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY is not set. Check .env")

    rf = Roboflow(api_key=api_key)

    # ↓あなたのRoboflow snippetに合わせて修正（例）
    project = rf.workspace("pd3-9x7yk").project("futsal-filed-detection-joozt")
    version = project.version(24)
    dataset = version.download("yolov8", location="./data")

    print("Done:", dataset.location)

if __name__ == "__main__":
    main()