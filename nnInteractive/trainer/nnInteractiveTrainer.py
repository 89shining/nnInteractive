import inspect
from typing import Union, Tuple, List

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from torch import nn


class nnInteractiveTrainer_stub():
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def build_network_architecture(plans_manager,
                                   dataset_json: dict,
                                   configuration_manager,
                                   num_input_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        # nnU-Net changed this API between releases: older versions require
        # ``arch_init_kwargs_req_import``, whereas newer versions no longer
        # accept it. Select arguments by the installed nnU-Net signature so an
        # inference bundle remains usable with either supported dependency.
        build = nnUNetTrainer.build_network_architecture
        parameters = inspect.signature(build).parameters
        if "plans_manager" in parameters:
            # nnU-Net >= 2.6 API. Its label manager determines the output
            # channels from dataset_json; only add nnInteractive's seven
            # interaction channels to the image inputs.
            kwargs = {
                "plans_manager": plans_manager,
                "dataset_json": dataset_json,
                "configuration_manager": configuration_manager,
                "num_input_channels": num_input_channels + 7,
                # nnInteractive checkpoints are trained with two CE logits
                # (background / foreground), even for a binary target.
                "num_output_channels": 2,
                "enable_deep_supervision": enable_deep_supervision,
            }
        else:
            # Legacy nnU-Net API, retained for reproducibility with older
            # released nnInteractive environments.
            kwargs = {
                "architecture_class_name": configuration_manager.network_arch_class_name,
                "arch_init_kwargs": configuration_manager.network_arch_init_kwargs,
                "arch_init_kwargs_req_import": configuration_manager.network_arch_init_kwargs_req_import,
                "num_input_channels": num_input_channels + 7,
                "num_output_channels": 2,
                "enable_deep_supervision": enable_deep_supervision,
            }
        return build(**{key: value for key, value in kwargs.items() if key in parameters})
