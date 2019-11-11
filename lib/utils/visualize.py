""" visualization tools 
"""


import torchvision
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tensorboardX import SummaryWriter

writer = SummaryWriter(save_path)

inputs, classes = next(iter(dataset.val_loader))
imgs = torchvision.utils.make_grid(inputs, nrow=8, padding=5, normalize=True)
writer.add_image('valdata_snapshot', imgs)




def vis_confusion(writer, step, matrix, class_dict):
    """
    Visualization of confusion matrix

    Parameters:
        writer (tensorboard.SummaryWriter): TensorBoard SummaryWriter instance.
        step (int): Counter usually specifying steps/epochs/time.
        matrix (numpy.array): Square-shaped array of size class x class.
            Should specify cross-class accuracies/confusion in percent
            values (range 0-1).
        class_dict (dict): Dictionary specifying class names as keys and
            corresponding integer labels/targets as values.
    """

    all_categories = sorted(class_dict, key=class_dict.get)

    # Normalize by dividing every row by its sum
    matrix = matrix.astype(float)
    for i in range(len(class_dict)):
        matrix[i] = matrix[i] / matrix[i].sum()

    # Create the figure
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Show the matrix and define a discretized color bar
    cax = ax.matshow(matrix)
    fig.colorbar(cax, boundaries=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

    # Set up axes. Rotate the x ticks by 90 degrees.
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # Turn off the grid for this plot. Enforce a tight layout to reduce white margins
    ax.grid(False)
    plt.tight_layout()

    # Call our auxiliary to TensorBoard function to render the figure 
    plot_to_tensorboard(writer, fig, step)



def plot_to_tensorboard(writer, fig, step):
    """
    Takes a matplotlib figure handle and converts it using
    canvas and string-casts to a numpy array that can be
    visualized in TensorBoard using the add_image function

    Parameters:
        writer (tensorboard.SummaryWriter): TensorBoard SummaryWriter instance.
        fig (matplotlib.pyplot.fig): Matplotlib figure handle.
        step (int): counter usually specifying steps/epochs/time.
    """

    # Draw figure on canvas
    fig.canvas.draw()

    # Convert the figure to numpy array, read the pixel values and reshape the array
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Normalize into 0-1 range for TensorBoard(X). Swap axes for newer versions where API expects colors in first dim
    img = img / 255.0
    # img = np.swapaxes(img, 0, 2) # if your TensorFlow + TensorBoard version are >= 1.8

    # Add figure in numpy "image" to TensorBoard writer
    writer.add_image('confusion_matrix', img, step)
    plt.close(fig)