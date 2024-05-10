import cv2
from utils.helper import GetLogger#, Predictor
from utils.main import Predictor
from argparse import ArgumentParser, Namespace
import sys
import numpy as np

# parser = ArgumentParser()
# parser.add_argument(
#     "--input", type=str, help="Set the input path to the video", required=True
# )
# parser.add_argument(
#     "--out", type=str, help="Set the output path to the video", required=True
# )
# args = parser.parse_args()

def convert(args):
    logger = GetLogger.logger(__name__)
    predictor = Predictor(args.path_to_model, args.path_to_weights)

    # Open video file
    video_path = args.input
    cap = cv2.VideoCapture(video_path)

    # Get video information
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    logger.info(f"No of frames {n_frames}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create VideoWriter for output
    output_path = args.out
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process each frame
    done = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # debug
        # batch_size = 5
        # frame = np.tile(frame[np.newaxis, ...], (batch_size, 1, 1, 1))
        out_frame, out_frame_seg = predictor.predict(frame)

        # Write the frame to the output video
        out.write(out_frame_seg)

        done += 1
        percent = int((done / n_frames) * 100)
        sys.stdout.write(
            "\rProgress: [{}{}] {}%".format("=" * percent, " " * (100 - percent), percent)
        )
        sys.stdout.flush()

    # Release video capture and writer
    cap.release()
    out.release()

if __name__=='__main__':
    args = Namespace()
    args.input = './assets/01646-video2.mp4'
    args.out = './assets/01646-video2_seg.mp4'
    convert(args)
    print('Done')
