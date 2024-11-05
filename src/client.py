import numpy as np
import flwr as fl
from src.blockchain import W3, dataset_contract
from src.utils import DEVICE, get_parameters, set_parameters, train

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import flwr as fl
from flwr.common import (
    Code,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from lime import lime_image

from src.visualization import plot_saliency_maps, plot_weight_adjustments


class XAIFederatedClient(fl.client.Client):
    def __init__(self, cid, net, trainloader, valloader):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.adjust_model_weights = plot_saliency_maps
        self.plot_weight_adjustments = plot_weight_adjustments
        self.optimizer = optim.Adam(self.net.parameters())
        self.lime_explainer = lime_image.LimeImageExplainer()
        print(f"Client {self.cid} initialized with network architecture.")
        self.training_history = {"loss": [], "accuracy": [], "weight_adjustments": []}

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        return GetParametersRes(
            status=Status(Code.OK, "Success"),
            parameters=ndarrays_to_parameters(get_parameters(self.net)),
        )

    def set_parameters(self, parameters: List[np.ndarray]):
        set_parameters(self.net, parameters)

    def refine_model_with_xai(self):
        images, labels = next(iter(self.trainloader))
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        lime_exp = self.lime_explainer.explain_instance(
            images[0].cpu().numpy().transpose(1, 2, 0),
            lambda x: self.net(torch.Tensor(x).to(DEVICE).permute(0, 3, 1, 2))
            .detach()
            .cpu()
            .numpy(),
            top_labels=5,
            hide_color=0,
            num_samples=100,
        )

        self.net.eval()
        images.requires_grad_()
        outputs = self.net(images)
        outputs.backward(torch.ones_like(outputs))
        saliency_maps = images.grad.abs().sum(dim=1)

        self.plot_saliency_maps(images, saliency_maps)

        original_weights = list(self.net.parameters())
        self.prioritize_data_streams(lime_exp, saliency_maps)
        self.adjust_model_weights(lime_exp, saliency_maps)

        adjusted_weights = list(self.net.parameters())
        self.plot_weight_adjustments(original_weights, adjusted_weights)

    def display_lime_explanation(self, image, lime_exp):
        temp = lime_exp.get_image_and_mask(
            lime_exp.top_labels[0],
            positive_only=False,
            num_features=10,
            hide_rest=False,
        )

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(image.transpose(1, 2, 0))  # Transpose to HWC format for display
        ax[0].set_title("Original Image")
        ax[1].imshow(temp)
        ax[1].set_title("LIME Explanation")
        plt.show()

    def fit(self, ins: FitIns) -> FitRes:
        print(f"[Client {self.cid}] Starting fit")
        set_parameters(self.net, parameters_to_ndarrays(ins.parameters))

        initial_weights = [param.clone().detach() for param in self.net.parameters()]
        loss_log, accuracy_log = train(
            self.net,
            self.trainloader,
            epochs=50,
            optimizer=self.optimizer,
            client_id=self.cid,
            dataset_contract=dataset_contract, 
            W3=W3
        )

        final_weights = [param.clone().detach() for param in self.net.parameters()]
        weight_adjustments = [
            torch.norm(f - i).item() for f, i in zip(final_weights, initial_weights)
        ]

        self.training_history["loss"].extend(loss_log)
        self.training_history["accuracy"].extend(accuracy_log)
        self.training_history["weight_adjustments"].append(np.mean(weight_adjustments))

        parameters = get_parameters(self.net)
        return FitRes(
            status=Status(Code.OK, "Success"),
            parameters=ndarrays_to_parameters(parameters),
            num_examples=len(self.trainloader.dataset),
            metrics={
                "loss": loss_log[-1],
                "accuracy": accuracy_log[-1],
                "weight_adjustment": np.mean(weight_adjustments),
            },
        )

    def prioritize_data_streams(self, lime_exp, saliency_maps):
        important_features = self.get_important_features(lime_exp, saliency_maps)
        for name in self.net.named_parameters():
            if any(feature in name for feature in important_features):
                self.optimizer.param_groups[0][
                    "lr"
                ] *= 1.1  # Increase learning rate by 10%

    def adjust_model_weights(self, saliency_maps):
        for i, (name, param) in enumerate(self.net.named_parameters()):
            if "weight" in name:
                saliency_importance = torch.mean(saliency_maps).item()
                param.data *= (
                    1 + 0.1 * saliency_importance
                )  # Increase weights by up to 10% based on saliency importance

    def get_important_features(self, lime_exp, saliency_maps):
        lime_importance = lime_exp.local_exp[lime_exp.top_labels[0]]
        saliency_importance = saliency_maps.view(-1).cpu().numpy()
        combined_importance = np.mean([lime_importance, saliency_importance], axis=0)
        return [
            f"feature_{i}" for i in np.argsort(combined_importance)[-5:]
        ]  # Top 5 important features
