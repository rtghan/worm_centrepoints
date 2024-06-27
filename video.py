import os
import matplotlib.pyplot as plt
import cv2
from worm import Worm
import subprocess
import glob
import itertools
from PIL import Image
from skimage.restoration import denoise_tv_bregman
import skimage.util

proj_dir = 'C:/Users/chane/Downloads/school files/uni/2023-2024/summer/ROP/pipelines/Kymograph_v1-main'
os.chdir(proj_dir)

from image import *
from process import *
from classes import *

##File parameters
import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda")
# device = torch.device("cuda:0")

def save_img(data, name='', i=0):
    if len(name) == 0:
        name = "data"

    new_p = Image.fromarray(data)
    if new_p.mode != 'RGB':
        new_p = new_p.convert('RGB')

    ending = "%02d.png" % i
    formatted_name = name + ending
    new_p.save(formatted_name)
def get_masked_video(data_dir: str, fname: str, save: str, init_head_pos: list[int], start: int, end: int, segnet,
                     denoise: float = 2, thresh: int = 30) -> Worm:
    """
    Takes a video saved at fpath, runs the processing pipeline on it, and saves the masked & eroded result
    to a video at the location specified by save. The number of frames in the video to be processed can be
    specified by duration, and is by default 100.
    """
    # setup the video saving
    fps = 11
    size = (1024, 1024)
    out = cv2.VideoWriter(save, cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)

    # read in each frame from the video
    cap = cv2.VideoCapture(f'{data_dir}/{fname}')
    i = 0
    # head_guess = [253, 604]
    worm = Worm(fname, init_head_pos, False, segnet)

    # go through frame by frame
    while (cap.isOpened() and i < end):
        ret, frame = cap.read()
        if ret and i >= start:
            # crop the frame to only the worm
            vid_frame = np.asarray(frame)
            vid_frame = vid_frame[:, :, 0]
            height, width = vid_frame.shape
            split = width // 2
            vid_frame = np.array(vid_frame[:, split:])

            # put the frame through pipeline and save to video
            print(f"processing frame {i}")
            skeleton = worm.add_frame(vid_frame)

            if type(skeleton) != int:
                # draw out where the head and body points are
                head_guess = worm.head_positions[-1]
                v = 255
                # print(worm.body_points[-1])
                # for point in worm.body_points[-1]:
                #     try:
                #         for r, c in itertools.combinations_with_replacement(range(-2, 3), 2):
                #             skeleton[int(point[0]) + r][int(point[1]) + c] = v
                #     except IndexError:
                #         pass
                try:
                    for r, c in itertools.product(range(-3, 4), repeat=2):
                        skeleton[head_guess[0] + r][head_guess[1] + c] = 255
                except IndexError:
                    pass
                out.write(skeleton)
                print(f'processed frame {i}')

            elif skeleton == 1:
                print("User skipped frame.")
            elif skeleton == -1:
                print("Could not resolve head, skipping frame. ")

        i += 1
        cv2.destroyAllWindows()

    subprocess.call([
        'ffmpeg', '-framerate', '8', '-i', 'temp_processed_frames/file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
        'spline_points.mp4'
    ])
    subprocess.call([
        'ffmpeg', '-framerate', '8', '-i', 'temp_processed_frames/sorted_body%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
        'sorted_body_vid.mp4'
    ])
    os.chdir(f'{data_dir}/temp_processed_frames')
    for file_name in glob.glob("*.png"):
        os.remove(file_name)
    os.chdir(data_dir)

    cap.release()
    out.release()
    return worm


#Run parameters
model_dir = 'C:/Users/chane/Downloads/school files/uni/2023-2024/summer/ROP/models'
os.chdir(model_dir)
segnet = SegNet()
segnet.load_state_dict(torch.load("ver15.pth",map_location="cuda:0"))
segnet.to(device)
segnet.eval()

data_dir = 'C:/Users/chane/Downloads/school files/uni/2023-2024/summer/ROP/data'
os.chdir(data_dir)#\\Bad alignment\\h5_20220922_3 6000")

subprocess.call([
    'ffmpeg', '-framerate', '8', '-i', 'temp_processed_frames/file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
    'spline_points_stable_vid.mp4'
])

subprocess.call([
    'ffmpeg', '-framerate', '8', '-i', 'temp_processed_frames/sorted_body%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
    'sorted_body_vid.mp4'
])

os.chdir(f'{data_dir}/temp_processed_frames')
for file_name in glob.glob("*.png"):
    os.remove(file_name)
os.chdir(data_dir)

fp="recording_04242024_143948_10mins.avi" # r"E:\Behaviour 23\New folder\recording_04242024_135452_15minutes.avi"
output_name = 'stable'
start_frame = 0
end_frame = 300
initial_head_position = [811, 201]

# worm = get_masked_video(data_dir, fp, f'{output_name}.mp4', initial_head_position, start_frame, end_frame, segnet)
# # worm = get_masked_video(data_dir, fp, 'test.mp4', 398, 420)
# # print(f'shortest segments: {worm.min_points}')
# # worm.to_csv()
# worm.runtime_to_csv()
# worm.position_csv.close()
