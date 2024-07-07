from pathlib import Path

from autoencoders.utils import *
from autoencoders.autoencoder import *

X_autoencoder_path = Path("./autoencoders/").resolve()

def load_X_autoencoder(model_ae, encoder_path, decoder_path):
    """
    :param X_autoencoder:
        object of class Autoencoder
    :param encoder_path:
    :param decoder_path:
    :return:
    """

    model_ae.encoder.load_state_dict(torch.load(encoder_path))
    model_ae.decoder.load_state_dict(torch.load(decoder_path))
