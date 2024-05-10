import os
import numpy as np
from numpy.lib.function_base import disp
import torch
import decord
from PIL import Image
from torchvision import transforms
# from random_erasing import RandomErasing
import warnings
from decord import VideoReader, cpu
from torch.utils.data import Dataset
# import video_transforms as video_transforms 
# import volume_transforms as volume_transforms
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_tensor
import detectron2.data.transforms as T
from glob import glob

class VideoDataset(torch.utils.data.Dataset):
    """Load your own video classification dataset.
    Parameters
    ----------
    root : str, required.
        Path to the root folder storing the dataset.
    setting : str, required.
        A text file describing the dataset, each line per video sample.
        There are three items in each line: (1) video path; (2) video length and (3) video label.
    train : bool, default True.
        Whether to load the training or validation set.
    test_mode : bool, default False.
        Whether to perform evaluation on the test set.
        Usually there is three-crop or ten-crop evaluation strategy involved.
    name_pattern : str, default None.
        The naming pattern of the decoded video frames.
        For example, img_00012.jpg.
    video_ext : str, default 'mp4'.
        If video_loader is set to True, please specify the video format accordinly.
    is_color : bool, default True.
        Whether the loaded image is color or grayscale.
    modality : str, default 'rgb'.
        Input modalities, we support only rgb video frames for now.
        Will add support for rgb difference image and optical flow image later.
    num_segments : int, default 1.
        Number of segments to evenly divide the video into clips.
        A useful technique to obtain global video-level information.
        Limin Wang, etal, Temporal Segment Networks: Towards Good Practices for Deep Action Recognition, ECCV 2016.
    num_crop : int, default 1.
        Number of crops for each image. default is 1.
        Common choices are three crops and ten crops during evaluation.
    new_length : int, default 1.
        The length of input video clip. Default is a single image, but it can be multiple video frames.
        For example, new_length=16 means we will extract a video clip of consecutive 16 frames.
    new_step : int, default 1.
        Temporal sampling rate. For example, new_step=1 means we will extract a video clip of consecutive frames.
        new_step=2 means we will extract a video clip of every other frame.
    temporal_jitter : bool, default False.
        Whether to temporally jitter if new_step > 1.
    video_loader : bool, default False.
        Whether to use video loader to load data.
    use_decord : bool, default True.
        Whether to use Decord video loader to load data. Otherwise use mmcv video loader.
    transform : function, default None.
        A function that takes data and label and transforms them.
    data_aug : str, default 'v1'.
        Different types of data augmentation auto. Supports v1, v2, v3 and v4.
    lazy_init : bool, default False.
        If set to True, build a dataset instance without loading any dataset.
    """
    def __init__(self,
                 setting,
                 root='',
                 cfg=None,
                #  train=True,
                #  test_mode=False,
                #  name_pattern='img_%05d.jpg',
                 video_ext='mp4',
                #  is_color=True,
                #  modality='rgb',
                #  num_segments=1,
                #  num_crop=1,
                #  new_length=1,
                #  new_step=1,
                #  transform=None,
                #  temporal_jitter=False,
                 video_loader=True,
                 use_decord=False,
                 lazy_init=False,
                 exclude_list=None
                 ):

        super(VideoDataset, self).__init__()
        self.root = root
        self.setting = setting
        self.exclude_list = exclude_list

        self.video_loader = video_loader
        self.video_ext = video_ext
        self.use_decord = use_decord
        self.lazy_init = lazy_init
        if not cfg is None:
            self.aug = T.ResizeShortestEdge(
                [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
            )
        else:
            self.aug = None

        if not self.lazy_init:
            self.clips = self._make_dataset(root, setting, exclude_list)
            if len(self.clips) == 0:
                raise(RuntimeError("Found 0 video clips in subfolders of: " + root + "\n"
                                   "Check your data directory (opt.data-dir)."))

    def __getitem__(self, index):
        try:
            directory = self.clips[index]
            if self.video_loader:
                if '.' in directory.split('/')[-1]:
                    # data in the "setting" file already have extension, e.g., demo.mp4
                    video_name = directory
                else:
                    # data in the "setting" file do not have extension, e.g., demo
                    # So we need to provide extension (i.e., .mp4) to complete the file name.
                    video_name = '{}.{}'.format(directory, self.video_ext)
                
        

                decord_vr = decord.VideoReader(video_name, num_threads=1)
                duration = len(decord_vr)
                fps = decord_vr.get_avg_fps()


            # segment_indices, skip_offsets = self._sample_train_indices(duration)
            
            images = self._video_batch_loader(video_name, decord_vr, duration)

            # # process_data, mask = self.transform((images, None)) # T*C,H,W
            # process_data = process_data.view((self.new_length, 3) + process_data.size()[-2:]).transpose(0,1)  # T*C,H,W -> T,C,H,W -> C,T,H,W
            
            # return (process_data, mask)
            # conver to torch
            images = self.transform(images)
            # images = torch.stack([to_tensor(img) for img in images])
        except Exception as e:
            print(f'====Problem with file {video_name}====')
            print(e)
            images, video_name, fps = [], [], []
        return images, video_name, fps
    def transform(self, images):
        # resize
        images = np.array(images)
        # images = images.transpose(0, 3, 1, 2)
        if not self.aug is None:
            
            resizer = self.aug.get_transform(images[0])
            images = [resizer.apply_image(img) for img in images]
        
        # conver to torch
        images = [to_tensor(img) for img in images]
        images = torch.stack(images).permute(0, 2,3,1)
        return images
    
    def __len__(self):
        return len(self.clips)

    def _make_dataset(self, directory, setting, exclude_list_path):
        # setting can be a path to folder with all the files
        exclude_list = []
        if not exclude_list_path is None:
            if not os.path.exists(exclude_list_path):
                raise(RuntimeError("Setting file %s doesn't exist. Check opt.train-list and opt.val-list. " % (setting)))
            exclude_list = []
            with open(exclude_list_path) as split_f:
                data = split_f.readlines()
                for line in data:
                    clip_path = os.path.join(directory, line.strip())
                    clip_path = clip_path.replace('densepose', 'clips')
                    # item = clip_path
                    exclude_list.append(clip_path)

        if os.path.isdir(setting):
            clips = glob(os.path.join(setting, self.video_ext))
            print(f'Loading from folder {setting}; {len(clips)} fils were found')
        else:
            if not os.path.exists(setting):
                raise(RuntimeError("Setting file %s doesn't exist. Check opt.train-list and opt.val-list. " % (setting)))
            clips = []
            with open(setting) as split_f:
                data = split_f.readlines()
                for line in data:
                    clip_path = os.path.join(directory, line.strip())
                    if clip_path in exclude_list:
                        continue
                    item = clip_path
                    clips.append(item)
        return clips

    # def _sample_train_indices(self, num_frames):
    #     average_duration = (num_frames - self.skip_length + 1) // self.num_segments
    #     if average_duration > 0:
    #         offsets = np.multiply(list(range(self.num_segments)),
    #                               average_duration)
    #         offsets = offsets + np.random.randint(average_duration,
    #                                               size=self.num_segments)
    #     elif num_frames > max(self.num_segments, self.skip_length):
    #         offsets = np.sort(np.random.randint(
    #             num_frames - self.skip_length + 1,
    #             size=self.num_segments))
    #     else:
    #         offsets = np.zeros((self.num_segments,))

    #     if self.temporal_jitter:
    #         skip_offsets = np.random.randint(
    #             self.new_step, size=self.skip_length // self.new_step)
    #     else:
    #         skip_offsets = np.zeros(
    #             self.skip_length // self.new_step, dtype=int)
    #     return offsets + 1, skip_offsets


    def _video_batch_loader(self, directory, video_reader, duration):
        sampled_list = []
        # frame_id_list = []
        # for seg_ind in indices:
        #     offset = int(seg_ind)
        #     for i, _ in enumerate(range(0, self.skip_length, self.new_step)):
        #         if offset + skip_offsets[i] <= duration:
        #             frame_id = offset + skip_offsets[i] - 1
        #         else:
        #             frame_id = offset - 1
        #         frame_id_list.append(frame_id)
        #         if offset + self.new_step < duration:
        #             offset += self.new_step
        frame_id_list = range(duration)
        try:
            video_data = video_reader.get_batch(frame_id_list).asnumpy()
            sampled_list = [Image.fromarray(video_data[vid, :, :, :]).convert('RGB') for vid, _ in enumerate(frame_id_list)]
        except:
            raise RuntimeError('Error occured in reading frames {} from video {} of duration {}.'.format(frame_id_list, directory, duration))
        return sampled_list
