import matplotlib.pyplot as plt
import numpy as np
import torchvision

def imshow(img, title=None):
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    if title:
        plt.title(title)
    plt.show()

def show_batch(loader, classes):
    data_iter = iter(loader)
    images, labels = next(data_iter)
    imshow(torchvision.utils.make_grid(images))
    print(' '.join(f'{classes[labels[j]]}' for j in range(len(labels))))

