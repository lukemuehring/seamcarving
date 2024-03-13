import numpy as np
import scipy as sp
import cv2
import scipy.signal                     # option for a 2D convolution library
from matplotlib import pyplot as plt    # for optional plots
import copy


"""Seam Carving Project

References
----------
(1) "Seam Carving for Content-Aware Image Resizing"
    Avidan and Shamir, 2007
    
(2) "Improved Seam Carving for Video Retargeting"
    Rubinstein, Shamir and Avidan, 2008
    
    
FUNCTIONS
----------
    IMAGE GENERATION:
        beach_backward_removal
        dolphin_backward_insert + with redSeams=True
        dolphin_backward_5050
        bench_backward_removal + with redSeams=True
        bench_forward_removal + with redSeams=True
        car_backward_insert
        car_forward_insert
    COMPARISON METRICS:
        difference_image
        numerical_comparison
"""

def getEnergyMap(img):
    # energy = |dx| + |dy|
    sobel_blue_X = cv2.Sobel(img[:, :, 0], cv2.CV_64F, 1, 0, ksize=5    , borderType=cv2.BORDER_REFLECT_101)
    sobel_green_X = cv2.Sobel(img[:, :, 1], cv2.CV_64F, 1, 0, ksize=5, borderType=cv2.BORDER_REFLECT_101)
    sobel_red_X = cv2.Sobel(img[:, :, 2], cv2.CV_64F, 1, 0, ksize=5, borderType=cv2.BORDER_REFLECT_101)

    sobel_blue_Y = cv2.Sobel(img[:, :, 0], cv2.CV_64F, 0, 1, ksize=5, borderType=cv2.BORDER_REFLECT_101)
    sobel_green_Y = cv2.Sobel(img[:, :, 1], cv2.CV_64F, 0, 1, ksize=5, borderType=cv2.BORDER_REFLECT_101)
    sobel_red_Y = cv2.Sobel(img[:, :, 2], cv2.CV_64F, 0, 1, ksize=5, borderType=cv2.BORDER_REFLECT_101)

    dx = np.square(sobel_blue_X) + np.square(sobel_green_X) + np.square(sobel_red_X)
    dy = np.square(sobel_blue_Y) + np.square(sobel_green_Y) + np.square(sobel_red_Y)

    return np.add(dx, dy)

def calculateCost(M):
    cost = np.copy(M)
    directions = np.zeros_like(M)
    rows, cols = cost.shape[0:2]
    max_value = np.finfo(M.dtype).max
    for i in range(1, rows):
        for j in range(cols):
            if (j == cols - 1):
                neighbor_list = [cost[i - 1][j - 1], cost[i - 1][j], max_value]
            elif (j == 0):
                neighbor_list = [max_value, cost[i - 1][j], cost[i - 1][j + 1]]
            else:
                neighbor_list = [cost[i - 1][j - 1], cost[i - 1][j], cost[i - 1][j + 1]]

            min_idx = np.argmin(neighbor_list)
            local_min = neighbor_list[min_idx]

            previous_value = cost[i][j]

            cost[i][j] = previous_value + local_min
            directions[i][j] = min_idx - 1

    return cost, directions


def findMinimalSeam(cost, directions):
    rows, cols = cost.shape[0:2]
    seam_idxs = np.zeros(rows)
    current_col_idx = int(np.argmin(cost[rows - 1]))

    for i in range(rows - 1, -1, -1):
        seam_idxs[i] = current_col_idx
        next = directions[i][current_col_idx]
        current_col_idx = int(min(max(next + current_col_idx, 0), cols - 1))

    return seam_idxs.astype(int)

def removeSeam(img, seam_idxs0, is2D=False):
    seam_idxs = np.copy(seam_idxs0)
    rows, cols = img.shape[0:2]

    mask = np.zeros((rows, cols)).flatten()

    row_idxs = np.arange(rows) * cols

    seam_idxs += row_idxs
    mask[seam_idxs] = 1.0
    mask = mask.reshape(rows, cols)

    if is2D:
        return img[mask == 0].reshape(rows, cols - 1)
    else:
        return img[mask == 0].reshape(rows, cols - 1, 3)

def insert_seam(img, seam_idxs0, redSeam=False):
    seam_idxs = np.copy(seam_idxs0)
    rows, cols = img.shape[0:2]
    expanded_img = np.zeros((rows, cols + 1, 3))
    for i in range(rows):
        current_col = seam_idxs[i]
        for color_channel in range(3):
            if redSeam:
                if color_channel == 2:
                    pixel = 255.
                else:
                    pixel = 0
                expanded_img[i, :current_col, color_channel] = img[i, :current_col, color_channel]
                # insert pixel (0, 0, 255)
                expanded_img[i, current_col, color_channel] = pixel
                expanded_img[i, current_col + 1:, color_channel] = img[i, current_col:, color_channel]
            else:
                if (current_col == 0):
                    averaged_pixel = np.mean(img[i, 0:2, color_channel])
                    expanded_img[i, 0, color_channel] = img[i, 0, color_channel]
                    # insert averaged pixel in between neighbors
                    expanded_img[i, 1, color_channel] = averaged_pixel
                    expanded_img[i, 2:, color_channel] = img[i, 1:, color_channel]
                else:
                    # average left and right, insert between
                    averaged_pixel = np.mean(img[i, current_col - 1: current_col + 1, color_channel])
                    expanded_img[i, :current_col, color_channel] = img[i, :current_col, color_channel]
                    # insert averaged pixel in between neighbors
                    expanded_img[i, current_col, color_channel] = averaged_pixel
                    expanded_img[i, current_col + 1:, color_channel] = img[i, current_col:, color_channel]

    return expanded_img


def color_seams(img, mask):
    rows, cols = img.shape[0:2]
    red = np.zeros((rows, cols, 3), np.float64)
    red[:] = (0, 0, 255)

    # Set values untouched by seam removal to be the same as the final image
    for i in range(rows):
        red[i][mask[i]] = img[i][mask[i]]

    return red

def pixel_diff(pixel1, pixel2):
    return np.abs(pixel1 - pixel2)

# -------------------------------------------------------------------
""" IMAGE GENERATION
    Parameters and Returns are as follows for all of the removal/insert 
    functions:

    Parameters
    ----------
    image : numpy.ndarray (dtype=uint8)
        Three-channel image of shape (r,c,ch)
    pctSeams : float
        Decimal value in range between(0. - 1.); percent of vertical seams to be
        inserted or removed.
    redSeams : boolean
        Boolean variable; True = this is a red seams image, False = no red seams
        
    Returns
    -------
    numpy.ndarray (dtype=uint8)
        An image of shape (r, c_new, ch) where c_new = new number of columns.
"""

def backward_removal(image, pctSeams=.50, redSeams=False):
    # setup
    img = np.copy(image).astype(np.float64)
    if redSeams:
        original = np.copy(img)

    rows, cols = img.shape[0:2]
    num_cols_to_remove = int(pctSeams * cols)

    # if we need to draw red seams, create an array to save the indexes
    if redSeams:
        col_indexes = np.indices((rows, cols))[1]

    # delete later
    # start_time = time.time()
    # current_time = time.time()

    for i in range(num_cols_to_remove):
        # calculate energy map each time using energy function or sobel filter
        energyMap = getEnergyMap(img)
        # print("---energy map %s seconds ---" % (time.time() - current_time))
        # current_time = time.time()

        # calculate cost of each pixel
        cost, directions = calculateCost(energyMap)
        # print("---calculateCost %s seconds ---" % (time.time() - current_time))
        # current_time = time.time()

        # find seam with minimal cost from image
        seam_idxs = findMinimalSeam(cost, directions)
        # print("---findMinimalSeam %s seconds ---" % (time.time() - current_time))
        # current_time = time.time()

        # remove seam
        img = removeSeam(img, seam_idxs)
        # print("---removeSeam %s seconds ---" % (time.time() - current_time))
        # current_time = time.time()

        if redSeams:
            col_indexes = removeSeam(col_indexes, seam_idxs, is2D=True)

        # print("one iteration")

    if redSeams:
        return color_seams(original, col_indexes)
    else:
        return img

def backward_insert(image, pctSeams=.50, redSeams=False):
    # setup
    img = np.copy(image).astype(np.float64)
    expanded_img = np.copy(image).astype(np.float64)

    if redSeams:
        original = np.copy(img)

    rows, cols = img.shape[0:2]
    num_cols_to_insert = int(np.ceil(pctSeams * cols))

    list_of_seams = []

    # if we need to draw red seams, create an array to save the indexes
    if redSeams:
        col_indexes = np.indices((rows, cols))[1]

    # delete later
    # start_time = time.time()
    # current_time = time.time()

    for i in range(num_cols_to_insert):
        # calculate energy map each time using energy function or sobel filter
        energyMap = getEnergyMap(img)
        # print("---energy map %s seconds ---" % (time.time() - current_time))
        # current_time = time.time()

        # calculate cost of each pixel
        cost, directions = calculateCost(energyMap)
        # print("---calculateCost %s seconds ---" % (time.time() - current_time))
        # current_time = time.time()

        # find seam with minimal cost from image
        seam_idxs = findMinimalSeam(cost, directions)
        # print("---findMinimalSeam %s seconds ---" % (time.time() - current_time))
        # current_time = time.time()

        #save seam indexes
        list_of_seams.append(seam_idxs)

        # remove seam
        img = removeSeam(img, seam_idxs)
        # print("---removeSeam %s seconds ---" % (time.time() - current_time))
        # current_time = time.time()

        # print("one iteration")

    for seam in range(num_cols_to_insert):
        current_seam_indexes = list_of_seams[seam]
        expanded_img = insert_seam(expanded_img, current_seam_indexes, redSeams)

    return expanded_img

def forward_removal(image, pctSeams=.50, redSeams=False):
    # setup
    img = np.copy(image).astype(np.float64)
    if redSeams:
        original = np.copy(img)

    rows, cols = img.shape[0:2]
    num_cols_to_remove = int(pctSeams * cols)

    # if we need to draw red seams, create an array to save the indexes
    if redSeams:
        col_indexes = np.indices((rows, cols))[1]

    # delete later
    # start_time = time.time()
    # current_time = time.time()

    for i in range(num_cols_to_remove):

        # calculate cost of each pixel
        cost, directions = calculateCostForwardEnergy(img)
        # print("---calculateCost %s seconds ---" % (time.time() - current_time))
        # current_time = time.time()

        # find seam with minimal cost from image
        seam_idxs = findMinimalSeam(cost, directions)
        # print("---findMinimalSeam %s seconds ---" % (time.time() - current_time))
        # current_time = time.time()

        # remove seam
        img = removeSeam(img, seam_idxs)
        # print("---removeSeam %s seconds ---" % (time.time() - current_time))
        # current_time = time.time()

        if redSeams:
            col_indexes = removeSeam(col_indexes, seam_idxs, is2D=True)

        # print("one iteration")

    if redSeams:
        return color_seams(original, col_indexes)
    else:
        return img

def forward_insert(image, pctSeams=.50, redSeams=False):
    # setup
    img = np.copy(image).astype(np.float64)
    expanded_img = np.copy(image).astype(np.float64)

    if redSeams:
        original = np.copy(img)

    rows, cols = img.shape[0:2]
    num_cols_to_insert = int(np.ceil(pctSeams * cols))

    list_of_seams = []

    # if we need to draw red seams, create an array to save the indexes
    if redSeams:
        col_indexes = np.indices((rows, cols))[1]

    # delete later
    # start_time = time.time()
    # current_time = time.time()

    for i in range(num_cols_to_insert):

        # calculate cost of each pixel
        cost, directions = calculateCostForwardEnergy(img)
        # print("---calculateCost %s seconds ---" % (time.time() - current_time))
        # current_time = time.time()

        # find seam with minimal cost from image
        seam_idxs = findMinimalSeam(cost, directions)
        # print("---findMinimalSeam %s seconds ---" % (time.time() - current_time))
        # current_time = time.time()

        #save seam indexes
        list_of_seams.append(seam_idxs)

        # remove seam
        img = removeSeam(img, seam_idxs)
        # print("---removeSeam %s seconds ---" % (time.time() - current_time))
        # current_time = time.time()


    for seam in range(num_cols_to_insert):
        current_seam_indexes = list_of_seams[seam]
        expanded_img = insert_seam(expanded_img, current_seam_indexes, redSeams)

    return expanded_img

def calculateCostForwardEnergy(img):

    avg_pixel_intensities = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    avg_pixel_intensities = avg_pixel_intensities.astype(np.float64)

    rows, cols = img.shape[0:2]

    base_energy = np.zeros_like(avg_pixel_intensities)
    cum_min_energy = np.zeros_like(avg_pixel_intensities)
    directions = np.zeros_like(avg_pixel_intensities)

    # https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array
    left_neighbors = np.roll(avg_pixel_intensities, 1, 1)
    top_neighbors = np.roll(avg_pixel_intensities, 1, 0)
    right_neighbors = np.roll(avg_pixel_intensities, -1, 1)

    cost_if_continue_up = np.abs(right_neighbors - left_neighbors)
    cost_if_continue_left = cost_if_continue_up + np.abs(top_neighbors - left_neighbors)
    cost_if_continue_right = cost_if_continue_up + np.abs(top_neighbors - right_neighbors)

    for current_row in range(0, rows):
        if current_row == 0:
            cum_min_energy[0] = cost_if_continue_up[0]
            # directions[0] is already defaulted at 0, to go upwards

        else:
            energy_previous_row = cum_min_energy[current_row - 1]
            total_cost_left = np.roll(energy_previous_row, 1) + cost_if_continue_left[current_row]
            total_cost_up = energy_previous_row + cost_if_continue_up[current_row]
            total_cost_right = np.roll(energy_previous_row, -1) + cost_if_continue_right[current_row]

            cost_options = np.array([total_cost_left, total_cost_up, total_cost_right])
            min_indexes = np.argmin(cost_options, axis=0)
            cum_min_energy[current_row] = np.choose(min_indexes, cost_options)
            directions[current_row] = min_indexes - 1 #subtract to keep directions in range {-1, 0 ,1}

    return cum_min_energy, directions


def beach_backward_removal(image, pctSeams=0.50, redSeams=False):
    """ Uses the backward method of seam carving from the 2007 paper to remove
    50% of the vertical seams in the provided image.
    """
    return backward_removal(image, pctSeams, redSeams).astype(np.uint8)

def dolphin_backward_insert(image, pctSeams=0.50, redSeams=False):
    """ Fig 8c, 8d from 2007 paper. Uses the backward method of seam carving to insert
    vertical seams in the image.
    
    This function is called twice:  Fig 8c with red seams
                                    Fig 8d without red seams
    """
    return backward_insert(image, pctSeams, redSeams).astype(np.uint8)


def dolphin_backward_5050(image, pctSeams=0.50, redSeams=False):
    """ Fig 8f from 2007 paper. Uses the backward method of seam carving to insert
    vertical seams in the image.
    
    *****************************************************************
    IMPORTANT NOTE: this function is passed the image array from the 
    dolphin_backward_insert function in main.py
    *****************************************************************
    
    """
    width = image.shape[1]
    original_width = width / (1.0 + pctSeams)
    final_width = original_width * 2
    new_pctSeams = final_width / width - 1.0

    return backward_insert(image, new_pctSeams, redSeams).astype(np.uint8)



def bench_backward_removal(image, pctSeams=0.50, redSeams=False):
    """ Fig 8 from 2008 paper. Uses the backward method of seam carving to remove
    vertical seams in the image.
    
    This function is called twice:  Fig 8 backward with red seams
                                    Fig 8 backward without red seams
    """
    return backward_removal(image, pctSeams, redSeams).astype(np.uint8)


def bench_forward_removal(image, pctSeams=0.50, redSeams=False):
    """ Fig 8 from 2008 paper. Uses the forward method of seam carving to remove
    vertical seams in the image.
    
    This function is called twice:  Fig 8 forward with red seams
                                    Fig 8 forward without red seams
  """
    return forward_removal(image, pctSeams, redSeams).astype(np.uint8)


def car_backward_insert(image, pctSeams=0.50, redSeams=False):
    """ Fig 9 from 2008 paper. Uses the backward method of seam carving to insert
    vertical seams in the image.
    """
    return backward_insert(image, pctSeams, redSeams).astype(np.uint8)


def car_forward_insert(image, pctSeams=0.50, redSeams=False):
    """ Fig 9 from 2008 paper. Uses the backward method of seam carving to insert
    vertical seams in the image.
    """
    return forward_insert(image, pctSeams, redSeams).astype(np.uint8)

# __________________________________________________________________
""" COMPARISON METRICS 
    There are two functions here, one for visual comparison support and one 
    for a quantitative metric."""

def difference_image(result_image, comparison_image):
    """ Takes two images and produce a difference image that best visually
    indicates where the two images differ in pixel values.
    
    Parameters
    ----------
    result_image, comparison_image : numpy.ndarray (dtype=uint8)
        two BGR images of the same shape (r,c,ch) to be compared
    
    Returns
    -------
    numpy.ndarray (dtype=uint8)
        An image of shape (r, c, ch) representing the difference between two
        images.
    """
    img1 = np.copy(result_image).astype(np.float64)
    img2 = np.copy(comparison_image).astype(np.float64)

    diff = cv2.subtract(img1, img2)
    diff[:][:][0] = 0
    diff[:][:][2] = 0
    diff[:][:][1] = np.abs(diff[:][:][1])
    norm = np.zeros((diff.shape[0],diff.shape[0]))


    diff = cv2.normalize(diff, norm, 0, 255, cv2.NORM_MINMAX)

    return diff.astype(np.uint8)

def numerical_comparison(result_image, comparison_image):
    """ Takes two images and produce one or two single-value metrics that
    numerically best indicate(s) how different or similar two images are.

    Parameters
    ----------
    result_image, comparison_image : numpy.ndarray (dtype=uint8)
        two BGR images of the same shape (r,c,ch) to be compared

    Returns
    -------
    value(s) : float
        One or two single_value metric comparisons
    """
    
    img1 = np.copy(result_image).astype(np.float64)
    img2 = np.copy(comparison_image).astype(np.float64)
    return ssim(img1, img2)


def ssim(img_x, img_y):
    # https://medium.com/srm-mic/all-about-structural-similarity-index-ssim-theory-code-in-pytorch-6551b455541e

    k1 = .01
    k2 = .03
    l = 255.
    c1 = (k1*l)**2
    c2 = (k2*l)**2

    # Making a Gaussian distribution
    filter = np.zeros((11, 11))
    middle = filter.shape[0] // 2
    filter[middle][middle] = 1.

    filter = cv2.GaussianBlur(filter, (11, 11), 0)

    mu_x = cv2.filter2D(img_x, -1, filter)
    mu_y = cv2.filter2D(img_y, -1, filter)

    mu_x_squared = mu_x**2
    mu_y_squared = mu_y**2

    sigma_x_squared = cv2.filter2D(img_x**2, -1, filter) - mu_x_squared
    sigma_y_squared = cv2.filter2D(img_y**2, -1, filter) - mu_y_squared

    sigma_xy = cv2.filter2D(img_x * img_y, -1, filter) - mu_x * mu_y

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denom = (mu_x_squared + mu_y_squared + c1) * (sigma_x_squared + sigma_y_squared + c2)

    ssim_metric = numerator / denom

    return np.mean(ssim_metric)

if __name__ == "__main__":
    pass
