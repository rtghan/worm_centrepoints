import numpy as np
import itertools
from image import *
from PIL import Image
import matplotlib.pyplot as plt
from skimage.restoration import denoise_tv_bregman
import skimage.util
import cv2
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import CubicSpline
import csv
import sys
from datetime import datetime
from time import process_time

class Worm:
    """
    Class to model the worm and keep track of its body. "Skeleton points" refers to the points that make up the centre
    line of the worm basically.

    Creation Parameters:
    src: The name of the file of the video recording of this worm.
    intial_head_guess: a (row, column) coordinate detailing an approximate guess for the head position on the
                       first frame
    skip: Whether or not to automatically skip problematic frames
    segnet: The CNN that we are using to segment the video frames.

    Attributes:
        src: The name of the file the worm comes from
        fps: The fps of the video (used for generating timestamps in the .csv)
        body_points: a list of the evenly spaced body points along the centreline of the worm, starting at the head
                     at each frame
        sorted_body: a list of the sorted points of the worm centreline from head to end at each frame
        min_points: the minimum number of body points detected across all frames
        max_points: the max number of body points detecte across all frames
        skip: whether or not to automatically skip error-throwing frames
        segnet: the CNN used to segment the worm frame
        runtime: a list of the runtimes of each component of the pipeline for each frame
    """
    def __init__(self, src, initial_head_guess, skip, segnet):
        self.src = src
        if initial_head_guess is not None:
            self.head_positions = [initial_head_guess]
        self.fps = 11
        self.body_points = []
        self.sorted_body = []
        self.min_points = 9999
        self.max_points = -1
        self.skip = skip
        self.segnet = segnet
        self.runtime = []

    def add_frame(self, video_frame, denoise: float = 2, thresh: int = 30, spacing: int = 15):
        """
        Update the information of the worm with a new frame. Returns 1 if the user chose to skip the frame on the first
        frame, -1 for the user choosing to skip a frame in general.
        """
        # the base augmentation is denoise -> histogram equalization -> thresholding
        denoise_start = process_time()
        augment_frame = skimage.util.img_as_ubyte(denoise_tv_bregman(video_frame, weight=denoise))
        denoise_end = process_time()

        # histogram equalization
        augment_frame = cv2.equalizeHist(augment_frame)
        hist_end = process_time()

        # fast thresholding using numpy
        thresh_frame = np.asarray(augment_frame)
        thresh_indices = thresh_frame > thresh
        thresh_frame[thresh_indices] = 255 # slow: use double for loop manual thresholding
        thresh_end = process_time()

        # segment the frame
        cnn_start = process_time()
        skeleton_frame = self.get_mask(thresh_frame)
        cnn_end = process_time()

        # attempt to grab the head
        head_grab_start = process_time()
        ret = self.get_head(skeleton_frame)
        head_grab_end = process_time()

        # handle the case when it is the first frame and user chose to skip to next frame
        if ret == 1:
            return 1

        # the case when the head tracking reported an error due to the new head position being too far from the old one
        elif ret == -1:
            # try again with the non-thresholded frame, which seems to be more stable, albeit more prone to body spikes
            print("Thresholded frame errored, trying again with less augmented video frame...")
            self.save_img(thresh_frame, "fail_thresh", len(self.head_positions))
            skeleton_frame = self.get_mask(augment_frame)
            ret = self.get_head(skeleton_frame, first_try=False)

        # if there is an error and the user still wants to skip, then we must return
        if ret == -1:
            self.save_img(augment_frame, "fail_augment", len(self.head_positions))
            return -1

        # otherwise proceed with the rest of the body update
        # self.get_skeleton(skeleton_frame)
        body_sort_start = process_time()
        self.body_sort(skeleton_frame)
        body_sort_end = process_time()

        # get body points using interpolation
        interp_start = process_time()
        self.get_skeleton(spacing)

        # select the x and y coordinates respectively to plot
        f_x_vals, f_y_vals = np.asarray(self.body_points[-1]).T
        interp_end = process_time()

        file_save_start = process_time()
        plt.plot(f_x_vals, f_y_vals, '.', alpha=0.9)
        # plt.plot(x_vals, y_vals, '-r', alpha=0.5)
        ax = plt.gca()
        ax.set_xlim([0, 1024])
        ax.set_ylim([0, 1024])

        plt.savefig("temp_processed_frames/file%02d.png" % len(self.head_positions), dpi=300)
        plt.clf()
        file_save_end = process_time()

        # track runtime of each component
        times = [(denoise_end - denoise_start, "denoise"), (hist_end - denoise_end, "hist"),
                 (thresh_end - hist_end, "thresh"), (cnn_end - cnn_start, "cnn"),
                 (head_grab_end - head_grab_start, "get head"), (body_sort_end - body_sort_start, "body_sort"),
                 (interp_end - interp_start, "interp"), (file_save_end - file_save_start, "file_save")]
        times_dict = {stage: time for time, stage in times}
        self.runtime.append(times_dict)

        print(f'Runtime of pipeline parts: {times}')

        return thresh_frame

    def get_mask(self, original_arr: np.array):
        """
        Takes a frame from the video, and puts it through the pipeline to get the segmented/skeletonized version.
        """
        mask = getPrediction(original_arr, self.segnet, torch.device("cuda"))
        mask = ChooseLargestBlob(mask)
        # mask_filled = fillHoles(mask)
        mask_erode = erode(mask)
        mask_erode = ChooseLargestBlob(mask_erode)
        mask_skeleton = skeletonize(mask_erode)

        return mask_skeleton

    def get_interp(self, ERROR_TOL=20, method=UnivariateSpline):
        """
        Attempts to interpolate the points that make up the body of the worm from the latest frame
        into a smooth function, determined by method. By default, UnivariateSpline.
        """
        x_vals = []
        y_vals = []
        dists = []
        cdist = 0
        incr_dst = 0
        t_vals = []
        skip = 1

        for i in range(1, (len(self.sorted_body[-1]) - 1) // skip):
            # as long as the gap between subsequent points is not too long, add it to be fitted by the spline
            if incr_dst <= ERROR_TOL:
                x_vals.append(self.sorted_body[-1][skip * i][1])
                y_vals.append(1024 - self.sorted_body[-1][skip * i][0])
                dists.append(cdist)
                incr_dst = math.dist((self.sorted_body[-1][i]), (self.sorted_body[-1][i+1]))
                cdist += incr_dst
                t_vals.append(i)

        x_interp = method(dists, x_vals)
        y_interp = method(dists, y_vals)

        return [x_interp, y_interp, dists, x_vals, y_vals]

    def get_head(self, skeleton_frame, ERROR_TOL=80, first_try = True):
        """
        Input a new frame of skeleton points to the worm to update where its head is
        """

        # let user pick where the head starts at
        if len(self.head_positions) == 0:
            head_candidates = self._get_head_candidates(skeleton_frame)
            user_choice = self._get_user_selection(skeleton_frame, head_candidates)

            if user_choice == 'Skip_frame':
                return 1

            self.head_positions.append(user_choice)

        # otherwise we can assume that we have the head location of the previous frame
        else:
            head_candidates = self._get_head_candidates(skeleton_frame)

            # take the head candidate that is the closest to the previous head point
            prev = self.head_positions[-1]
            prev_head_distances = [(math.dist(prev, head_cand), head_cand) for head_cand in head_candidates]
            head_guess = sorted(prev_head_distances, key=lambda x: x[0])[0][1]

            # make sure the new guess is reasonable
            if math.dist(prev, head_guess) < ERROR_TOL:
                self.head_positions.append(head_guess)
            else:
                # if the first try returns an error on the head, then we should try again with a different base image
                if first_try:
                    return -1

                user_choice = self._get_user_selection(skeleton_frame, head_candidates)

                # stop updating the worm for this frame if the user desires (choosing 'Skip Frame')
                # if the user's choice is not "Skip Frame", then save the frame, otherwise, return -1
                if type(user_choice) != str:
                    self.head_positions.append(user_choice)
                else:
                    return -1

        return 0

    def _get_user_selection(self, skeleton_frame, head_candidates) -> list[int, int]:
        """
        Given an input frame of skeleton points, generates the possible head candidates, and lets the user
        choose which one should be designated head.
        """
        sorted_skeleton = sortSkeleton(skeleton_frame)
        head_candidates.append(sorted_skeleton[0])
        head_candidates.append(self.head_positions[-1])
        head_candidates.append("Skip frame")
        head_candidate = None

        # show user options
        for i in range(len(head_candidates)):
            if type(head_candidates[i]) == str:
                print(f'{i}: {head_candidates[i]}')
            else:
                print(f'{i}: (row, column) - ({head_candidates[i][0]}, {head_candidates[i][1]}) ')
        # TODO: show user a picture of the messed up image
        mod = skeleton_frame
        for i in range(len(head_candidates) - 1):
            try:
                for r, c in itertools.product(range(-3, 4), repeat=2):
                    mod[head_candidates[i][0] + r][head_candidates[i][1] + c] = 255
            except IndexError:
                pass

        new_p = Image.fromarray(mod)
        if new_p.mode != 'RGB':
            new_p = new_p.convert('RGB')
        new_p.show()

        # get input
        while head_candidate is None:
            try:
                selection = int(input("Select the head/option: "))
                head_candidate = head_candidates[selection]
            except (IndexError, ValueError) as e:
                print("Invalid selection, try again. ")
                pass

        return head_candidate

    def _get_head_candidates(self, skeleton_frame) -> list[list[int, int]]:
        """
        Given an input frame of skeleton points, outputs a list of possible head points.
        """
        N_ROWS, N_COLS = skeleton_frame.shape

        # for each point in the skeleton, keep track of the vectors that point to the closest few points
        vec_closest = []
        skeleton_points = np.column_stack(np.where(skeleton_frame != 0))

        # set a search radius around each point for the points nearby it
        RADIUS = 1 # slow: 5
        steps = list(itertools.product(range(-RADIUS, RADIUS + 1), repeat=2))

        for point in skeleton_points:
            # grab any nearby points
            nearby_points = []
            for row_step, col_step in steps:
                search_row, search_col = point[0] + row_step, point[1] + col_step
                if 0 <= search_row < N_ROWS and 0 <= search_col < N_COLS:
                    if skeleton_frame[search_row][search_col] == 255:
                        nearby_points.append((search_row, search_col))

            # compute the vectors that go from the point to its nearby neighbours
            closest = [np.subtract(n, point) for n in nearby_points if not np.array_equal(n, point)]
            vec_closest.append((point, closest))

        # for non-head points, there should be vectors from it to neighbours that point in different directions
        # as the vectors should be angled more than 90 degrees apart, their dot product should be negative
        # hence any point that satisfies this cannot be a head candidate
        head_candidates = []
        for point, vectors in vec_closest:
            head_candidate = True
            for a, b in itertools.combinations(range(len(vectors)), 2):
                if np.dot(vectors[a], vectors[b]) < 0:
                    head_candidate = False
                    break # slow: no break

            if head_candidate:
                head_candidates.append(point)

        return head_candidates

    def body_sort(self, skeleton_frame):
        """
        Saves a sorted list of the body points in order of how close they are to the head (arc-length wise).
        """
        N_ROWS, N_COLS = skeleton_frame.shape
        self.sorted_body.append([])

        self.time = 0
        self.queued = np.zeros(skeleton_frame.shape)
        self.visited = np.zeros(skeleton_frame.shape)
        self.start_time = np.zeros(skeleton_frame.shape)
        self.end_time = np.zeros(skeleton_frame.shape)

        # TODO: Fix the dfs/body sorting, for some reason at the end we get random points from the middle?
        sys.setrecursionlimit(2000)
        self._dfs(self.head_positions[-1], skeleton_frame)
        sys.setrecursionlimit(1000)
        self.sorted_body[-1].reverse()
        self.save_sorted_body()

    def _dfs(self, curr_point, skeleton_frame):
        N_ROWS, N_COLS = skeleton_frame.shape
        self.time += 1
        r, c = curr_point[0], curr_point[1]

        # update node
        self.start_time[r][c] = self.time
        self.queued[r][c] = 1

        # set a search radius around each point for the points nearby it
        RADIUS = 1
        steps = list(itertools.product(range(-RADIUS, RADIUS + 1), repeat=2))

        # get the nearby points in the worm
        close_points = []
        for row_step, col_step in steps:
            search_row, search_col = r + row_step, c + col_step
            if 0 <= search_row < N_ROWS and 0 <= search_col < N_COLS:
                if (skeleton_frame[search_row][search_col] == 255) and (self.queued[search_row][search_col] == 0):
                    self.queued[search_row][search_col] = 1
                    p = [search_row, search_col]
                    close_points.append([math.dist(curr_point, p), p])

        # visit neighbours in order of closeness
        if len(close_points) > 0:
            sorted_close = sorted(close_points, key=lambda x: x[0])
            for dist, point in sorted_close:
                if self.visited[point[0]][point[1]] == 0:
                    self._dfs(point, skeleton_frame)

        self.visited[r][c] = 1
        self.time += 1
        self.end_time[r][c] = self.time
        self.sorted_body[-1].append(curr_point)

    def get_skeleton(self, spacing):
        """
        Generates body points from the latest stored position of the head that are approximately spacing units apart,
        arc-length wise
        """
        [f_x, f_y, t_vals, x_vals, y_vals] = self.get_interp()

        # use the spline to get approx. equally spaced points (arc-length) wise
        spaced_body_points = [(f_x(0), f_y(0))]
        skeleton_points = [(f_x(0), f_y(0))]
        dt = 0.25
        t, end = 0 + dt, max(t_vals)
        segment_length = 0

        # travel the spline in little increments until we reach one segment length
        while t < end:
            curr_point = (f_x(t), f_y(t))
            prev_point = (skeleton_points[-1])
            step_len = math.dist(curr_point, prev_point)
            segment_length += step_len
            skeleton_points.append(curr_point)

            # check if we have reached one segment length
            if segment_length >= spacing:
                # compute any overshoot:
                overshoot = segment_length - spacing
                adjustment_vec = (overshoot / segment_length) * np.subtract(prev_point, curr_point)
                spaced_body_points.append(curr_point + adjustment_vec)
                segment_length = overshoot

            # move down the worm
            t += dt

        # update the # points metrics as appropriate
        if len(spaced_body_points) < self.min_points:
            self.min_points = len(spaced_body_points)
        if len(spaced_body_points) > self.max_points:
            self.max_points = len(spaced_body_points)

        self.body_points.append(spaced_body_points)



        # N_ROWS, N_COLS = skeleton_frame.shape
        # visited = np.zeros((N_ROWS, N_COLS))
        # n_visited = 1
        # body_points = [self.head_positions[-1]]
        # explore_points = [self.head_positions[-1]]
        # visited[self.head_positions[-1][0]][self.head_positions[-1][1]] = 1
        #
        # # set a search radius around each point for the points nearby it
        # RADIUS = 3
        # steps = list(itertools.product(range(-RADIUS, RADIUS + 1), repeat=2))
        #
        # # travel along the worm from the head for a certain segment length and place a body point at the end until
        # # we reach the desired amount of body points or we run out of worm to travel
        # segment_length = 0
        # while len(explore_points) > 0:
        #     curr_point = explore_points.pop()
        #
        #     # get the next point in the worm
        #     close_points = []
        #     for row_step, col_step in steps:
        #         search_row, search_col = curr_point[0] + row_step, curr_point[1] + col_step
        #         if 0 <= search_row < N_ROWS and 0 <= search_col < N_COLS:
        #             if (skeleton_frame[search_row][search_col] == 255) and (visited[search_row][search_col] == 0):
        #                 p = [search_row, search_col]
        #                 close_points.append([math.dist(curr_point, p), p])
        #
        #     if len(close_points) > 0:
        #         sorted_close = sorted(close_points, key=lambda x: x[0])
        #         next_point = sorted_close[0][1]
        #         explore_points.append(next_point)
        #         step_dist = math.dist(curr_point, next_point)
        #         segment_length += step_dist
        #
        #         # see if we have reached the appropriate segment length
        #         if segment_length >= spacing:
        #             # compute the overshoot
        #             overshoot = segment_length - spacing
        #             adjustment_vec = (overshoot / segment_length) * np.subtract(curr_point, next_point)
        #             body_points.append(next_point + adjustment_vec)
        #             segment_length = overshoot
        #
        #         visited[curr_point[0]][curr_point[1]] = 1
        #         n_visited += 1
        #
        # if len(body_points) < self.min_points:
        #     self.min_points = len(body_points)
        # if len(body_points) > self.max_points:
        #     self.max_points = len(body_points)
        #
        # print(f"found {len(body_points)} points")
        #
        # self.body_points.append(body_points)

    def to_csv(self, filename=""):
        """
        Converts the saved position data of the worm to a csv format apt for usage in
        Ruby's UCI pipeline (as of 17/06/2024)
        """
        if len(filename) == 0:
            time = datetime.now()
            filename = f'csv_data/{time.hour}_{time.minute}_{self.src}.csv'

        with open(filename, mode='w', newline='') as worm_data:
            writer = csv.writer(worm_data, delimiter=',', quotechar="|", quoting=csv.QUOTE_MINIMAL)

            writer.writerow(["Worm Data", "Center Points (um)"])
            writer.writerow(["Sequence", self.src])
            writer.writerow([])
            writer.writerow(["","Worm 1"])

            columns = ["Frame", "Time"]
            for i in range(1, self.max_points + 1):
                columns.append(f'{i}x')
                columns.append(f'{i}y')

            writer.writerow(columns)

            time = 0
            for i in range(0, len(self.body_points)):
                data = [str(i + 1), str(time)]
                for point in self.body_points[i]:
                    r, c = point
                    x = c
                    y = 1024 - r
                    data.append(str(x))
                    data.append(str(y))
                writer.writerow(data)
                time += 1/self.fps

    def runtime_to_csv(self, filename=""):
        """
        Converts the runtimes of each part of the pipeline for each frame into a csv
        """
        if len(filename) == 0:
            time = datetime.now()
            filename = f'csv_data/runtime{time.hour}_{time.minute}_{self.src}.csv'

        with open(filename, mode='w', newline='') as rt_data:
            writer = csv.writer(rt_data, delimiter=',', quotechar="|", quoting=csv.QUOTE_MINIMAL)
            components = list(self.runtime[0].keys())

            writer.writerow(["frame"] + components)

            for i in range(len(self.runtime)):
                columns = [self.runtime[i][label] for label in components]
                writer.writerow([str(i)] + columns)

    def save_img(self, data, name='', i=0):
        if len(name) == 0:
            name = "data"

        new_p = Image.fromarray(data)
        if new_p.mode != 'RGB':
            new_p = new_p.convert('RGB')

        ending = "%02d.png" % i
        formatted_name = name + ending
        new_p.save(formatted_name)

    def save_body_points(self, i):
        """
        Saves the segmented centre body points into an image.
        """
        frame = np.zeros((1024, 1024))

        for point in self.body_points[-1]:
            try:
                for r, c in itertools.combinations_with_replacement(range(-2, 3), 2):
                    frame[point[0] + r][point[1] + c] = 255
            except IndexError:
                pass

        new_p = Image.fromarray(frame)
        if new_p.mode != 'RGB':
            new_p = new_p.convert('RGB')
        new_p.save("temp_processed_frames/body_points%02d.png" % i)

    def save_sorted_body(self):
        """
        Saves the latest recorded body of the worm, where the value of each pixel is greater the closer you are to the
        head.
        """
        frame = np.zeros((1024, 1024))

        val = 255
        i = 0
        for point in self.sorted_body[-1]:
            try:
                for r, c in itertools.combinations_with_replacement(range(-2, 3), 2):
                    frame[point[0] + r][point[1] + c] = val - i*(155/(len(self.sorted_body[-1]) + 1))
            except IndexError:
                pass
            i += 1
        new_p = Image.fromarray(frame)
        if new_p.mode != 'RGB':
            new_p = new_p.convert('RGB')
        new_p.save("temp_processed_frames/sorted_body%02d.png" % len(self.sorted_body))

def save_head_guess(head_guess, i):
    frame = np.zeros((1024, 1024))
    try:
        for r, c in itertools.combinations_with_replacement(range(-2, 3), 2):
            frame[head_guess[0] + r][head_guess[1] + c] = 255
    except IndexError:
        pass
    new_p = Image.fromarray(frame)
    if new_p.mode != 'RGB':
        new_p = new_p.convert('RGB')
    new_p.save("file%02d.png" % i)


def save_head_candidates(head_candidates, i):
    frame = np.zeros((1024, 1024))
    for point in head_candidates:
        row, column = point
        frame[row][column] = 255
    new_p = Image.fromarray(frame)
    if new_p.mode != 'RGB':
        new_p = new_p.convert('RGB')
    new_p.save('headcand%02d.png' % i)
