'''
names :
    Oren Holthman , 209905207
    Ishay Post , 205415607
'''
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot as plt
import math
import os


def in_bounds(i, j, image):
    """
        Checks if a given pixel (i, j) is within the bounds of an image.

        Parameters:
        i (int): The row index of the pixel.
        j (int): The column index of the pixel.
        image (list): A 2D list representing an image, where each element is a pixel value.

        Returns:
        bool: True if the pixel (i, j) is within the bounds of the image, False otherwise.
    """
    return i in range(len(image)) and j in range(len(image[0]))



def find_same(x_pix, y_pix, t_im, f_im, win, sim_weight, dis_weight):
    """
    Finds the pixel in the gray target image that is most likely to have the same color
    as the original color of a pixel in the source image
    :param x_pix:  The row index of the current pixel.
    :param y_pix:  The column index of the current pixel.
    :param t_im:   The source image
    :param f_im:   The target image
    :param win:    The size of the window in which the function will search
    sim_weight(float): The weight to be given to the sim value in the calculation.
    dis_weight(float): The weight to be given to the dis value in the calculation.

    :return:       X and Y values of the pixel
    """
    same = [x_pix, y_pix]
    mn = t_im[x_pix][y_pix] + 254
    for i in range(win):
        frame = get_pixels_in_frame(f_im, x_pix, y_pix, i)
        for f in frame:
            if in_bounds(f[0], f[1], t_im):
                new = new_min(x_pix, y_pix, f[0], f[1], mn, t_im, f_im, sim_weight, dis_weight)
                if new < mn:
                    same = f
                    mn = new
    return same


def similarity(x_pix, y_pix, i, j, t_im, f_im):
    """
    Calculates the absolute difference between the color values of two pixels.

    Parameters:
    x_pix (int): The row index of the first pixel.
    y_pix (int): The column index of the first pixel.
    i (int): The row index of the second pixel.
    j (int): The column index of the second pixel.
    t_im (list): A 2D list representing an image, where each element is a pixel value.
    f_im (list): A 2D list representing another image, where each element is a pixel value.

    Returns:
    int: The absolute difference between the color values of the two pixels.
    """
    return abs(int(t_im[x_pix][y_pix]) - int(f_im[i][j]))

def distance(x_pix, y_pix, i, j):
    """
        Calculates the Euclidean distance between two points in a 2D space.

        Parameters:
        x_pix (int): The x-coordinate of the first point.
        y_pix (int): The y-coordinate of the first point.
        i (int): The x-coordinate of the second point.
        j (int): The y-coordinate of the second point.

        Returns:
        float: The Euclidean distance between the two points.
    """
    return math.sqrt((x_pix - i) ** 2 + (y_pix - j) ** 2)

def new_min(x_pix, y_pix, i, j, mn, t_im, f_im, sim_weight, dis_weight):
    """
        Calculates the new minimum value based on the distance and similarity
        between two pixels in two images.

        Parameters:
        x_pix (int): The x-coordinate of the first pixel in the target image.
        y_pix (int): The y-coordinate of the first pixel in the target image.
        i (int): The x-coordinate of the second pixel in the other image.
        j (int): The y-coordinate of the second pixel in the other image.
        mn (float): The current minimum value.
        t_im (list): The target image as a 2D list of pixel values.
        f_im (list): The other image as a 2D list of pixel values.
        sim_weight(float): The weight to be given to the sim value in the calculation.
        dis_weight(float): The weight to be given to the dis value in the calculation.

        Returns:
        If the calculation result is less than the minimum, it will be returned as the new minimum,
        if not, the old minimum will be returned
    """
    dis = (distance(x_pix, y_pix, i, j))
    sim = (similarity(x_pix, y_pix, i, j, t_im, f_im))
    new = abs(sim_weight * sim + dis_weight * dis)
    if new < mn:
        return new
    return mn

def restore_colors(to_values, gray_from_values, from_values, black_img, win = 25 , sim_weight = 1.0, dis_weight = 1.5):
    """
        Restore the colors of a grayscale image based on a corresponding color image.

        Parameters:
        to_values (list of lists): A 2D list representing the grayscale image to be colorized.
        gray_from_values (list of lists): A 2D list representing a grayscale version of the corresponding color image.
        from_values (list of lists): A 2D list representing the target  color image.
        black_img (list of lists): A 2D list representing the output image, which will be colorized using the corresponding color image.
        win (int): The size of the window to use when searching for matching pixels. default 25
        sim_weight(float): The weight to be given to the sim value in the calculation. Default 1.0
        dis_weight(float): The weight to be given to the dis value in the calculation. Default 1.5

        Returns:
        black_img (list of lists): A 2D list representing the colorized version of the input grayscale image.
        """
    for i in range(len(to_values)):
        for j in range(len(to_values[0])):
            x, y = find_same(i, j, to_values, gray_from_values, win, sim_weight, dis_weight)
            black_img[i][j] = from_values[x][y]
    return black_img

def plot_results(org_image, restored_img, target_im, target_im_type = ''):
    """
        Plot the original, restored and target images side by side.

        Parameters:
        org_image (numpy array): The original image.
        restored_img (numpy array): The restored image.
        target (PIL Image object): The target image.
        target_im_type (str): A string with the type of the target image. Default is empty string.

        Returns:
        None

        """
    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(7, 7))

    # Display the images on the subplots
    ax1[0].imshow(org_image, cmap='gray')
    ax1[1].imshow(target_im.convert('L'), cmap='gray')
    ax2[0].imshow(restored_img)
    ax2[1].imshow(target_im)

    # Set titles for the subplots
    ax1[0].set_title('original')
    ax1[1].set_title('grey ' + target_im_type)
    ax2[0].set_title('restore image')
    ax2[1].set_title(target_im_type)

    for ax in [*ax1, *ax2]:
        ax.axis('off')

    #Show the plot
    plt.show()


def get_pixels_in_frame(image, x, y, rec_size):
    """
    Returns a list of all the pixels in the frame of the rectangle that surrounds
    the pixel at (x, y) in the given image.
    """
    x1, y1, x2, y2 = get_surrounding_rectangle(image, x, y, rec_size)
    pixels = []
    for i in range(y1, y2 + 1):
        for j in range(x1, x2 + 1):
            if i == y1 or i == y2 or j == x1 or j == x2:
                pixels.append([i, j])
    return pixels

def get_surrounding_rectangle(image, x, y, rec_size):
    """
    Returns the rectangle that surrounds the pixel at (x, y) in the given image.
    The rectangle is represented as a tuple of (x1, y1, x2, y2), where (x1, y1) is
    the top-left corner of the rectangle and (x2, y2) is the bottom-right corner.
    """
    width = len(image[0])
    height = len(image)
    # Calculate the coordinates of the top-left and bottom-right corners of the rectangle
    x1 = max(0, x - rec_size)
    y1 = max(0, y - rec_size)
    x2 = min(width - rec_size, x + rec_size)
    y2 = min(height - rec_size, y + rec_size)
    return (x1, y1, x2, y2)

def create_average_image(set_dir, name):
    """
        Create an average image from a set of images with the same name.

        Parameters:
        set_dir (str): The directory path where the images are stored.
        name (str): The common name of the images.

        Returns:
        numpy array: The average image.

    """
    same_people_img = []
    for i in range(1, 20):
        # construct the relative path to the image
        image_path = os.path.join(set_dir, name + '.' + str(i) + '.jpg')

        # open the image
        image = Image.open(image_path)
        data = np.asarray(image)
        # data = data.flatten()
        same_people_img.append(data)

    # Take the mean of all images in same_people_img
    return np.mean(same_people_img, axis=0)


persons = ['cgboyc', 'cmkirk', 'djhugh', 'dmwest', 'gmwate', 'khughe', 'lejnno']
#persons = ['cgboyc', 'cmkirk'] 

# question 3b
#load the images and change them to np.array to be ready for pca

train_set = []
names_train_set = []

# get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# construct the relative path to the training set directory
training_set_dir = os.path.join(current_dir, "faces_sets", "training_set")
#training_set_dir = ADD YOUR PATH HERE

for name in persons:
    for i in range(1, 20):
        # construct the relative path to the image
        image_path = os.path.join(training_set_dir, name + '.' + str(i) + '.jpg')

        # open the image
        image = Image.open(image_path).convert('L')
        data = np.asarray(image)
        data = data.flatten()
        train_set.append(data)
        names_train_set.append(name + '.' + str(i) + '.jpg')
train_set = np.asarray(train_set)

# Creating a black image in the dimensions of the images we will restore.
# Restoring an image will be done by repainting that image
black_img = Image.open(os.path.join(training_set_dir, 'cgboyc' + '.' + str(1) + '.jpg'))
black_img = np.asarray(black_img).copy()
black_img.fill(0)

test_set = []
names_test_set = []

# construct the relative path to the training set directory
test_set_dir = os.path.join(current_dir, "faces_sets", "test_set")
#test_set_dir = ADD YOUR PATH HERE
for name in persons:
    # construct the relative path to the image
    image_path = os.path.join(test_set_dir, name + '.' + str(20) + '.jpg')

    # open the image
    image = Image.open(image_path).convert('L')

    data = np.asarray(image)
    data = data.flatten()
    test_set.append(data)
    names_test_set.append(name + '.' + str(20) + '.jpg')
test_set = np.asarray(test_set)

#question 3c
# Perform PCA
n_components = len(train_set)
pca = PCA(n_components=n_components, svd_solver="randomized", whiten=True)
X_train_pca = pca.fit_transform(train_set)
X_train_1d = X_train_pca.reshape((X_train_pca.shape[0], -1))


#question 3e
for name in persons:
    # construct the relative path to the image
    image_path = os.path.join(test_set_dir, name + '.' + str(20) + '.jpg')
    # Opens the image and converts it to gray levels
    image = Image.open(image_path).convert('L')

    # Image processing and conversion to PCA space
    image_array = np.asarray(image)
    data = image_array.flatten().reshape(1,-1)
    data_pca = pca.transform(data)

    # Finding the most similar image in the PCA space
    similar = cosine_similarity(data_pca,X_train_pca)
    max_similarity_index = np.argmax(similar[0])
    most_similar_im = Image.open(os.path.join(training_set_dir, names_train_set[max_similarity_index]))

    # Preparing the parameters for "restore_colors"
    grey_most_sim = most_similar_im.convert('L')
    grey_most_sim_val = np.asarray(grey_most_sim)
    most_sim_vals = np.asarray(most_similar_im)

    # Creating the restored image and displaying the results
    restored_img = restore_colors(image_array, grey_most_sim_val, most_sim_vals, black_img.copy())
    plot_results(image, restored_img, most_similar_im, target_im_type='most similar')


#question 3f
for name in persons:
    # construct the relative path to the image
    image_path = os.path.join(test_set_dir, name + '.' + str(20) + '.jpg')
    # Opens the image and converts it to gray levels
    image = Image.open(image_path).convert('L')

    # Creating the average image and creating a gray copy of it
    avg_image = create_average_image(training_set_dir, name)
    grey_avg_image = np.asarray(Image.fromarray(avg_image.astype('uint8')).convert('L'))

    # Creating the restored image and displaying the results
    restored_img = restore_colors(np.asarray(image), grey_avg_image, avg_image, black_img.copy())
    restored_img = restored_img / 255.0
    plot_results(image, restored_img, Image.fromarray(avg_image.astype('uint8')),target_im_type='average image')
