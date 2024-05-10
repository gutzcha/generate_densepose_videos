from dataset import VideoDataset
try:
    from accelerate import Accelerator
except:
    Accelerator = None

import logging
import torch
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from densepose import add_densepose_config
from densepose.vis.extractor import (
    DensePoseResultExtractor,
)
from densepose.vis.densepose_results import (
    DensePoseResultsFineSegmentationVisualizer as Visualizer,
)

import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode, LazyConfig
from detectron2.data import (
    MetadataCatalog,
)

from detectron2.modeling import build_model
import os
from tqdm import tqdm
import cv2
from argparse import ArgumentParser, Namespace

import sys

def save_np_as_video(nparray, filename='output_video.avi', fps=25):
    height, width, channels = nparray[0].shape
    batch_size = len(nparray)
    fps = int(fps)
    
    # fourcc = cv2.VideoWriter_fourcc(*"H264")AVC1
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for f in range(batch_size):
        frame = nparray[f]
        out.write(frame)
    out.release()
class DefaultBatchPredictor(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format


    def forward(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (B,T, H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for batch of images.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                image = image[:, :, :, ::-1]
            b, height, width, c = image.shape

            image = image.float().permute(0,3,1,2)
            image.to(self.cfg.MODEL.DEVICE)
            inputs = [{"image": img, "height": height, "width": width} for img in image]
            # inputs = {"image": image, "height": height, "width": width}

            predictions = self.model(inputs)
            return predictions
    
   
class GetLogger:
    @staticmethod
    def logger(name):
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        return logging.getLogger(name)


class Predictor(torch.nn.Module):
    def __init__(self, path_to_model_config, path_to_weights):
        super().__init__()
        self.path_to_model_config = path_to_model_config
        self.path_to_weights = path_to_weights
        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(
            self.path_to_model_config
        )  # Use the path to the config file from DensePose
        cfg.MODEL.WEIGHTS = self.path_to_weights  # Use the path to the pre-trained model weights
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Adjust as needed
        # self.predictor = DefaultPredictor(cfg)
        self.predictor = DefaultBatchPredictor(cfg)
        self.extractor = DensePoseResultExtractor()
        self.visualizer = Visualizer()
        self.cfg = cfg

    def forward(self, frame):
        # is the input a batch of videos or one video with a batch of frames?
        if len(frame.shape) == 5:
            if frame.shape[0] == 1:
                frame = frame[0]
            else:
                raise ValueError("Batch of videos not supported")
        frame = frame*255
        with torch.no_grad():
            outputs = self.predictor(frame)#["instances"]
        outputs = [out["instances"] for out in outputs]
        # outputs = self.extractor(outputs)
        outputs = [self.extractor(out) for out in outputs]

        frame = frame.cpu().numpy().astype(np.uint8)
        out_frame = frame
        out_frame_seg = np.zeros_like(frame)

        [self.visualizer.visualize(of, ot) for of, ot in zip(out_frame, outputs)]
        [self.visualizer.visualize(of, ot) for of, ot in zip(out_frame_seg, outputs)]


        return (out_frame, out_frame_seg)


# Define a function to save progress to a text file
def save_progress(progress_data, file_path):
    with open(file_path, 'a') as f:
        f.write(progress_data)

def convert(args):
    # Initialize progress data
    progress_data = ""

    MAX_N_FRAMES = 16
    logger = GetLogger.logger(__name__)
    predictor = Predictor(args.path_to_model_config, args.path_to_weights)
    cfg = predictor.cfg
    # Open video file
    dataset = VideoDataset(setting=args.input, cfg=cfg,
                            root=args.root, exclude_list=args.exclude_list)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    if Accelerator is not None:
        accelerator = Accelerator()
        predictor, dataloader = accelerator.prepare(predictor, dataloader)

    n_files = len(dataloader)
    # Create VideoWriter for output
    output_path = args.out
    log_path = os.path.join(output_path, "log.txt")
    file_counter = 0
    for batch, filename, fps in tqdm(dataloader, total=n_files):
        file_counter += 1
        if len(batch) == 0:
            continue
        filename = filename[0]
        if output_path is None:
            # this_output_path = filename.replace('clip','densepose')
            this_output_path = filename.replace('color', 'densepose')
        else:
            basefilename = os.path.basename(filename)
            this_output_path = output_path + f'/{basefilename}'
        if args.verbose:
            print(f'Processing {filename} to {this_output_path}')
        # if os.path.exists(this_output_path):
        #     print(f'File {this_output_path} already exists')
        #     continue

        # Process each sub-batch
        out_frames = []
        out_frame_segs = []
        batch = batch[0]
        for frame in batch:
            # print(f'Frame shape before unsqueeze: {frame.shape}')
            frame = torch.unsqueeze(frame, 0)
            # print(f'Frame shape after unsqueeze: {frame.shape}')
            # Process sub-batch
            out_frame, out_frame_seg = predictor(frame)

            # print(f'Frame shape after out_frame_seg: {out_frame_seg.shape}')
            # print(f'Frame shape after out_frame: {out_frame.shape}')

            out_frames.append(out_frame)
            out_frame_segs.append(out_frame_seg)

        # Concatenate the outputs of sub-batches along the first axis
        # out_frame = np.concatenate(out_frames, axis=0)
        out_frame_seg = np.concatenate(out_frame_segs, axis=0)

        os.makedirs(os.path.dirname(this_output_path), exist_ok=True)
        save_np_as_video(out_frame_seg,this_output_path ,fps=fps)

        # Update progress data and save it to the file
        progress_data = f"Processed: {filename}  {file_counter} of {n_files}\n"
        save_progress(progress_data, log_path)

def parse_args():
    parser = ArgumentParser(description="Extract features from a video file")
    parser.add_argument("--input", type=str, default="/code/file_list.txt", help="path to video file list")
    parser.add_argument("--out", type=str, default=None, help="path to where to save videos")
    parser.add_argument("--path_to_model_config", type=str, default="/code/model_configs/densepose_rcnn_R_50_FPN_s1x.yaml", help="path to model config file")
    parser.add_argument("--path_to_weights", type=str, default="/code/models/model_final_162be9.pkl", help="path to model weights file")
    parser.add_argument("--exclude_list", type=str, default=None, help="path to video file list to exclude")
    parser.add_argument("--root", type=str, default="", help="add the root directory to the video file paths")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose mode")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    if args.verbose:
        print("Verbose mode enabled.")

    convert(args)
    print('=====================Done======================')
