class RiemannianAutoencoder:
    def __init__(self, euclidean, device):
        self.manifold = euclidean
        self.d = self.manifold.d
        self.eps = None
        self.d_eps = None
        self.device = device

    def encode(self, x):
        """
        :param x: N x [Epoint] tensor
        :return : N x d_eps tensor
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def decode(self, p):
        """
        :param a: N x d_eps tensor
        :return : N x [Epoint] tensor
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def project_on_manifold(self, x):
        """
        :param x: N x [Epoint] tensor
        :return : N x [Epoint] tensor
        """
        return self.decode(self.encode(x))