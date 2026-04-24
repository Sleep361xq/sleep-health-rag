class ConfigParser:
    """
    Minimal compatibility shim for loading legacy training checkpoints.

    The exported EEG sleep staging checkpoint stores a reference to
    `parse_config.ConfigParser` from the original training project.
    In the inference-only packaging we only need the class symbol to exist
    so PyTorch can deserialize the checkpoint container and extract the
    `state_dict`.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
