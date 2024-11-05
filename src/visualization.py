import os
import torch
import matplotlib.pyplot as plt
import numpy as np


# Create an images directory if it doesn't exist
if not os.path.exists('images'):
    os.makedirs('images')

def plot_saliency_maps(images, saliency_maps, client_id):
    """
    Plots and saves saliency maps for a given set of images.

    Args:
        images (torch.Tensor): Batch of input images.
        saliency_maps (torch.Tensor): Corresponding saliency maps.
        client_id (int): Identifier for the client.
    """
    images = images.cpu().numpy()
    saliency_maps = saliency_maps.cpu().detach().numpy()
    num_images = min(len(images), 5)  # Plot a maximum of 5 images
    plt.figure(figsize=(15, 5))

    for i in range(num_images):
        plt.subplot(2, num_images, i + 1)
        plt.imshow(images[i].transpose(1, 2, 0))  # Convert from CxHxW to HxWxC
        plt.title("Image")
        plt.axis('off')

        plt.subplot(2, num_images, i + 1 + num_images)
        plt.imshow(saliency_maps[i], cmap='hot')  # Use hot colormap for saliency maps
        plt.title("Saliency Map")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'images/saliency_maps_client_{client_id}.png')  # Save the figure
    plt.close()  # Close the figure to free memory


def plot_weight_adjustments(original_weights, adjusted_weights, client_id):
    """
    Plots histograms of the original and adjusted weights, saving them to disk.

    Args:
        original_weights (List[torch.Tensor]): Original model weights.
        adjusted_weights (List[torch.Tensor]): Adjusted model weights after training.
        client_id (int): Identifier for the client.
    """
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
    plt.savefig(f'images/weight_adjustments_client_{client_id}.png')  # Save the figure
    plt.close()  # Close the figure to free memory


def plot_federated_learning_metrics(strategy_instance):
    """
    Plots global federated learning metrics over rounds, including loss, accuracy,
    and average weight adjustments.

    Args:
        strategy_instance: Strategy instance containing the global history of metrics.
    """
    plt.figure(figsize=(15, 10))

    plt.subplot(3, 1, 1)
    plt.plot(strategy_instance.global_history['round'], strategy_instance.global_history['loss'])
    plt.title('Global Loss Over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Loss')

    plt.subplot(3, 1, 2)
    plt.plot(strategy_instance.global_history['round'], strategy_instance.global_history['accuracy'])
    plt.title('Global Accuracy Over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')

    plt.subplot(3, 1, 3)
    plt.plot(strategy_instance.global_history['round'], strategy_instance.global_history['weight_adjustments'])
    plt.title('Average Weight Adjustments Over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Weight Adjustment')

    plt.tight_layout()
    plt.savefig('images/federated_learning_metrics.png')  # Save the figure
    plt.close()


def analyze_results(strategy_instance):
    """
    Prints final federated learning results, including accuracy, loss, and average weight adjustment.

    Args:
        strategy_instance: Strategy instance containing the global history of metrics.
    """
    print("Federated Learning Results:")
    print(f"Final Accuracy: {strategy_instance.global_history['accuracy'][-1]:.4f}")
    print(f"Final Loss: {strategy_instance.global_history['loss'][-1]:.4f}")
    print(f"Average Weight Adjustment: {np.mean(strategy_instance.global_history['weight_adjustments']):.4f}")
