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
device = torch.device("cpu")
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

def get_mask(worm: Worm, original_arr: np.array, idx: int):
    """
    Takes a frame from the video, and puts it through the pipeline to get the segmented/skeletonized version.
    """
    mask = getPrediction(original_arr, segnet, torch.device("cpu"))
    mask = ChooseLargestBlob(mask)
    # mask_filled = fillHoles(mask)
    mask_erode = erode(mask)
    mask_erode = ChooseLargestBlob(mask_erode)
    mask_skeleton = skeletonize(mask_erode)
    if worm.add_frame(mask_skeleton) == -1:
        return -1

    # body_points = sortSkeleton(mask_skeleton)
    [f_x, f_y, t_vals, x_vals, y_vals] = worm.get_interp()
    # [spline_x, spline_y, t_vals, x_vals, y_vals] = fitSpline(worm.sorted_body[-1])
    # frame = np.zeros(np.shape(mask_skeleton))

    f_x_vals = f_x(t_vals)
    f_y_vals = f_y(t_vals)

    plt.plot(f_x_vals, f_y_vals, alpha=0.9)
    # plt.plot(x_vals, y_vals, '-r', alpha=0.5)
    ax = plt.gca()
    ax.set_xlim([0, 1024])
    ax.set_ylim([0, 1024])

    plt.savefig("file%02d.png" % idx, dpi=300)
    plt.clf()

    return mask_skeleton


def get_masked_video(data_dir: str, fname: str, save: str, init_head_pos: list[int], start: int, end: int, denoise: float = 2, thresh: int = 30) -> Worm:
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
    worm = Worm(fname, init_head_pos, False)

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
            data = skimage.util.img_as_ubyte(denoise_tv_bregman(vid_frame, weight=denoise))
            data = cv2.equalizeHist(data)
            data = threshold(data, thresh)

            # try and segment the current frame
            data = get_mask(worm, data, i)
            success = False
            if type(data) != int:
                success = True
            # if the neural network errored, try and segment a less processed frame
            elif not success:
                print("Trying less processed frame...")
                data = cv2.equalizeHist(vid_frame)
                data = get_mask(worm, data, i, first_try=False)

            if type(data) != int:
                # draw out where the head and body points are
                head_guess = worm.head_positions[-1]
                v = 255
                print(worm.body_points[-1])
                for point in worm.body_points[-1]:
                    try:
                        for r, c in itertools.combinations_with_replacement(range(-2, 3), 2):
                            data[int(point[0]) + r][int(point[1]) + c] = v
                    except IndexError:
                        pass
                try:
                    for r, c in itertools.product(range(-3, 4), repeat=2):
                        data[head_guess[0] + r][head_guess[1] + c] = 255
                except IndexError:
                    pass
                out.write(data)
                print(f'processed frame {i}')

        i += 1
        cv2.destroyAllWindows()

    subprocess.call([
        'ffmpeg', '-framerate', '8', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
        'spline.mp4'
    ])
    subprocess.call([
        'ffmpeg', '-framerate', '8', '-i', 'sorted_body%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
        'sorted_body_vid.mp4'
    ])
    for file_name in glob.glob("*.png"):
        os.remove(file_name)
    cap.release()
    out.release()
    return worm


#Run parameters
model_dir = 'C:/Users/chane/Downloads/school files/uni/2023-2024/summer/ROP/models'
os.chdir(model_dir)
segnet = SegNet()
segnet.load_state_dict(torch.load("ver15.pth",map_location=device))
segnet.eval()

data_dir = 'C:/Users/chane/Downloads/school files/uni/2023-2024/summer/ROP/data'
os.chdir(data_dir)#\\Bad alignment\\h5_20220922_3 6000")

subprocess.call([
    'ffmpeg', '-framerate', '8', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
    'spline.mp4'
])
subprocess.call([
    'ffmpeg', '-framerate', '8', '-i', 'sorted_body%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
    'sorted_body_vid.mp4'
])
for file_name in glob.glob("*.png"):
    os.remove(file_name)

fp="recording_04242024_135452_15minutes.avi" # r"E:\Behaviour 23\New folder\recording_04242024_135452_15minutes.avi"
output_name = 'test'
start_frame = 0
end_frame = 100
initial_head_position = [813, 190]

worm = get_masked_video(data_dir, fp, f'{output_name}.mp4', initial_head_position, start_frame, end_frame)
# worm = get_masked_video(data_dir, fp, 'test.mp4', 398, 420)
# # print(f'shortest segments: {worm.min_points}')
worm.to_csv()
