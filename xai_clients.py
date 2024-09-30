import torch
import numpy as np
from lime import lime_image
import shap

import matplotlib.pyplot as plt
import numpy as np

class XAIFederatedClient(fl.client.Client):
    def __init__(self, cid, net, trainloader, valloader):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.optimizer = optim.Adam(self.net.parameters())
        self.lime_explainer = lime_image.LimeImageExplainer()
        print(f"Client {self.cid} initialized with network architecture.")


    def refine_model_with_xai(self):
        # Get a batch of data
        images, labels = next(iter(self.trainloader))
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # LIME explanation
        lime_exp = self.lime_explainer.explain_instance(
            images[0].cpu().numpy().transpose(1, 2, 0),
            lambda x: self.net(torch.Tensor(x).to(DEVICE).permute(0, 3, 1, 2)).detach().cpu().numpy(),
            top_labels=5,
            hide_color=0,
            num_samples=100
        )

        # Alternative to SHAP: Using gradient-based saliency maps
        self.net.eval()
        images.requires_grad_()
        outputs = self.net(images)
        outputs.backward(torch.ones_like(outputs))
        saliency_maps = images.grad.abs().sum(dim=1)

        # Visualize saliency maps
        self.plot_saliency_maps(images, saliency_maps)

        # Save original weights for comparison
        original_weights = list(self.net.parameters())

        # Use XAI insights to refine the model
        self.prioritize_data_streams(lime_exp, saliency_maps)
        self.adjust_model_weights(lime_exp, saliency_maps)

        # Visualize weight adjustments
        adjusted_weights = list(self.net.parameters())
        self.plot_weight_adjustments(original_weights, adjusted_weights)


    def display_lime_explanation(self, image, lime_exp):
        # Convert the explanation to an image with mask
        temp, mask = lime_exp.get_image_and_mask(
            lime_exp.top_labels[0],
            positive_only=False,
            num_features=10,
            hide_rest=False
        )

        # Plot original image and LIME explanation side by side
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        # Display the original image
        ax[0].imshow(image.transpose(1, 2, 0))  # Transpose to HWC format for display
        ax[0].set_title("Original Image")

        # Display the LIME explanation
        ax[1].imshow(temp)
        ax[1].set_title("LIME Explanation")

        # Show the plots
        plt.show()

    def prioritize_data_streams(self, lime_exp, saliency_maps):
        # Example: Increase learning rate for features identified as important by LIME and saliency maps
        important_features = self.get_important_features(lime_exp, saliency_maps)
        for name, param in self.net.named_parameters():
            if any(feature in name for feature in important_features):
                self.optimizer.param_groups[0]['lr'] *= 1.1  # Increase learning rate by 10%

    def adjust_model_weights(self, lime_exp, saliency_maps):
        # Example: Adjust weights based on saliency maps
        for i, (name, param) in enumerate(self.net.named_parameters()):
            if 'weight' in name:
                saliency_importance = torch.mean(saliency_maps).item()
                param.data *= (1 + 0.1 * saliency_importance)  # Increase weights by up to 10% based on saliency importance

    def get_important_features(self, lime_exp, saliency_maps):
        lime_importance = lime_exp.local_exp[lime_exp.top_labels[0]]
        saliency_importance = saliency_maps.view(-1).cpu().numpy()
        combined_importance = np.mean([lime_importance, saliency_importance], axis=0)
        return [f"feature_{i}" for i in np.argsort(combined_importance)[-5:]]  # Top 5 important features
