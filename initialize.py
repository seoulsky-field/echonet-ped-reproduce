"""
The echonet package contains code for loading echocardiogram videos, and
functions for training and testing segmentation and ejection fraction
prediction models.
"""

import click

from version import __version__
from utils.config import CONFIG as config
import utils


@click.group()
def main():
    """Entry point for command line interface."""


del click


main.add_command(utils.segmentation.run)
main.add_command(utils.video.run)
main.add_command(utils.joint.run)

__all__ = ["__version__", "main", "utils"]
