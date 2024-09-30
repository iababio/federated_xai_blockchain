import matplotlib.pyplot as plt

def plot_saliency_maps(images, saliency_maps):
    # Convert images and saliency maps to numpy for plotting
    images = images.cpu().numpy()
    saliency_maps = saliency_maps.cpu().detach().numpy()

    num_images = min(len(images), 5)  # Plot a maximum of 5 images
    plt.figure(figsize=(15, 5))

    for i in range(num_images):
        ax = plt.subplot(2, num_images, i + 1)
        plt.imshow(images[i].transpose(1, 2, 0))  # Convert from CxHxW to HxWxC
        plt.title("Image")
        plt.axis('off')

        ax = plt.subplot(2, num_images, i + 1 + num_images)
        plt.imshow(saliency_maps[i].cpu(), cmap='hot')  # Use hot colormap for saliency maps
        plt.title("Saliency Map")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def plot_weight_adjustments(original_weights, adjusted_weights):
    # Flatten the weights for histogram
    original_flat = torch.cat([param.data.view(-1) for param in original_weights]).cpu().numpy()
    adjusted_flat = torch.cat([param.data.view(-1) for param in adjusted_weights]).cpu().numpy()

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(original_flat, bins=50, alpha=0.7, label='Original Weights', color='blue')
    plt.title("Original Weights Histogram")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(adjusted_flat, bins=50, alpha=0.7, label='Adjusted Weights', color='orange')
    plt.title("Adjusted Weights Histogram")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.legend()

    plt.tight_layout()
    plt.show()
