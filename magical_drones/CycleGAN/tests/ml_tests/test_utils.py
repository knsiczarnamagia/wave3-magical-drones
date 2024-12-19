import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from utils import Loss_visualise


def test_loss_visualise():
    loss_info = [
        {"Generator loss": 2.3, "Discriminator loss": 0.8},
        {"Generator loss": 2.1, "Discriminator loss": 0.9},
        {"Generator loss": 1.9, "Discriminator loss": 1.0},
        {"Generator loss": 1.7, "Discriminator loss": 1.1},
        {"Generator loss": 1.5, "Discriminator loss": 1.2},
        {"Generator loss": 1.3, "Discriminator loss": 1.3},
        {"Generator loss": 1.1, "Discriminator loss": 1.4},
        {"Generator loss": 0.9, "Discriminator loss": 1.5},
        {"Generator loss": 0.8, "Discriminator loss": 1.6},
    ]

    Loss_visualise(loss_info=loss_info, save_to="")


if __name__ == "__main__":
    test_loss_visualise()
