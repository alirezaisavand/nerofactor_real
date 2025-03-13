import torch


class Embedder:
    def __init__(
            self, incl_input=True, in_dims=3, log2_max_freq=3, n_freqs=4,
            log_sampling=True, periodic_func=None):
        """
        Positional encoding embedding class.

        Args:
            incl_input (bool): Whether to include the original input in the embedding.
            in_dims (int): Input dimensions.
            log2_max_freq (int): Log base 2 of the maximum frequency.
            n_freqs (int): Number of frequency bands.
            log_sampling (bool): Whether to use logarithmic sampling of frequencies.
            periodic_func (list of callable): Periodic functions (e.g., sin, cos).
        """
        if periodic_func is None:
            periodic_func = [torch.sin, torch.cos]

        self.embed_func = []  # List of embedding functions
        self.out_dims = 0  # Total output dimensions

        # Optionally include the input itself
        if incl_input:
            self.embed_func.append(lambda x: x)
            self.out_dims += in_dims

        # Generate frequency bands
        if log_sampling:
            freq_bands = 2. ** torch.linspace(0., log2_max_freq, n_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** log2_max_freq, n_freqs)

        # Create embedding functions for each frequency and periodic function
        for freq in freq_bands:
            for p_f in periodic_func:
                self.embed_func.append(
                    lambda x, p_f=p_f, freq=freq: p_f(x * freq))
                self.out_dims += in_dims

    def __call__(self, x):
        """
        Apply the embedding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape `(batch_size, in_dims)`.

        Returns:
            torch.Tensor: Embedded tensor of shape `(batch_size, out_dims)`.
        """
        return torch.cat([f(x) for f in self.embed_func], dim=-1)
