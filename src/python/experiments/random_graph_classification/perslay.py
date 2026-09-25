import numpy as np
import torch
from torch import nn


def permutation_equivariant_layer(inp, dimension, perm_op, lbda, b, gamma):
    """DeepSet PersLay."""
    num_pts = inp.shape[1]
    b = b.unsqueeze(0).unsqueeze(0)
    A = torch.einsum("ijk,kl->ijl", inp, lbda).reshape(-1, num_pts, dimension)

    if perm_op is not None:
        if perm_op == "max":
            beta = torch.max(inp, dim=1).values.unsqueeze(1).repeat(1, num_pts, 1)
        elif perm_op == "min":
            beta = torch.min(inp, dim=1).values.unsqueeze(1).repeat(1, num_pts, 1)
        elif perm_op == "sum":
            beta = torch.sum(inp, dim=1).unsqueeze(1).repeat(1, num_pts, 1)
        else:
            raise Exception("perm_op should be min, max or sum")

        B = torch.einsum("ijk,kl->ijl", beta, gamma).reshape(-1, num_pts, dimension)
        return A - B + b

    return A + b


def rational_hat_layer(inp, q, mu, r):
    """Rational Hat PersLay."""
    mu = mu.unsqueeze(0).unsqueeze(0)
    r = r.unsqueeze(0).unsqueeze(0)
    bc_inp = inp.unsqueeze(-1)
    norms = torch.linalg.vector_norm(bc_inp - mu, ord=q, dim=2)
    return 1 / (1 + norms) - 1 / (1 + torch.abs(torch.abs(r) - norms))


def rational_layer(inp, mu, sg, al):
    """Rational PersLay."""
    mu = mu.unsqueeze(0).unsqueeze(0)
    sg = sg.unsqueeze(0).unsqueeze(0)
    al = al.unsqueeze(0).unsqueeze(0)
    bc_inp = inp.unsqueeze(-1)
    return 1 / torch.pow(
        1 + torch.sum(torch.abs(bc_inp - mu) * torch.abs(sg), dim=2),
        al,
    )


def exponential_layer(inp, mu, sg):
    """Exponential PersLay."""
    mu = mu.unsqueeze(0).unsqueeze(0)
    sg = sg.unsqueeze(0).unsqueeze(0)
    bc_inp = inp.unsqueeze(-1)
    return torch.exp(
        torch.sum(
            -torch.square(bc_inp - mu) * torch.square(sg),
            dim=2,
        )
    )


def landscape_layer(inp, sp):
    """Landscape PersLay."""
    sp = sp.unsqueeze(0).unsqueeze(0)
    return torch.maximum(
        0.5 * (inp[:, :, 1:2] - inp[:, :, 0:1])
        - torch.abs(sp - 0.5 * (inp[:, :, 1:2] + inp[:, :, 0:1])),
        torch.zeros((), dtype=inp.dtype, device=inp.device),
    )


def betti_layer(inp, theta, sp):
    """Betti PersLay."""
    sp = sp.unsqueeze(0).unsqueeze(0)
    X, Y = inp[:, :, 0:1], inp[:, :, 1:2]
    return 1.0 / (
        1.0
        + torch.exp(
            -theta
            * (
                0.5 * (Y - X)
                - torch.abs(sp - 0.5 * (Y + X))
            )
        )
    )


def entropy_layer(inp, theta, sp):
    """Entropy PersLay.

    WARNING: this function assumes that padding values are zero.
    """
    sp = sp.unsqueeze(0).unsqueeze(0)
    transform = torch.tensor(
        [[1.0, -1.0], [0.0, 1.0]],
        dtype=inp.dtype,
        device=inp.device,
    )
    bp_inp = torch.einsum("ijk,kl->ijl", inp, transform)

    L = bp_inp[:, :, 1:2]
    X = bp_inp[:, :, 0:1]
    Y = bp_inp[:, :, 0:1] + bp_inp[:, :, 1:2]

    LN = L / torch.sum(L, dim=1, keepdim=True)
    entropy_terms = torch.where(LN > 0.0, -LN * torch.log(LN), LN)

    return entropy_terms / (
        1.0
        + torch.exp(
            -theta
            * (
                0.5 * (Y - X)
                - torch.abs(sp - 0.5 * (Y + X))
            )
        )
    )


def image_layer(inp, image_size, image_bnds, sg):
    """Persistence Image PersLay."""
    transform = torch.tensor(
        [[1.0, -1.0], [0.0, 1.0]],
        dtype=inp.dtype,
        device=inp.device,
    )
    bp_inp = torch.einsum("ijk,kl->ijl", inp, transform)

    dimension_before, num_pts = inp.shape[2], inp.shape[1]
    coords = [
        torch.arange(
            start=image_bnds[i][0],
            end=image_bnds[i][1],
            step=(image_bnds[i][1] - image_bnds[i][0]) / image_size[i],
            dtype=inp.dtype,
            device=inp.device,
        )
        for i in range(dimension_before)
    ]
    M = torch.meshgrid(*coords, indexing="xy")
    mu = torch.cat([tens.unsqueeze(0) for tens in M], dim=0)

    bc_inp = bp_inp.reshape(
        [-1, num_pts, dimension_before]
        + [1 for _ in range(dimension_before)]
    )

    return (
        torch.exp(
            torch.sum(
                -torch.square(bc_inp - mu) / (2 * torch.square(sg)),
                dim=2,
            )
        )
        / (2 * np.pi * torch.square(sg))
    ).unsqueeze(-1)


class PerslayModel(nn.Module):

    def __init__(self, name, diagdim, perslay_parameters, rho):
        super().__init__()
        self.namemodel = name
        self.diagdim = diagdim
        self.perslay_parameters = perslay_parameters
        self.rho = rho

        self.vars = []
        self._parameters_by_channel = nn.ModuleList()
        self._final_models = nn.ModuleDict()

        for nf, plp in enumerate(self.perslay_parameters):
            channel = nn.ParameterList()

            weight = plp["pweight"]
            if weight is not None:
                Winit, Wtrain = plp["pweight_init"], plp["pweight_train"]

                if not callable(Winit):
                    Wiv = Winit
                elif weight == "power":
                    Wiv = Winit([1])
                elif weight == "grid":
                    Wiv = Winit(plp["pweight_size"])
                elif weight == "gmix":
                    Wiv = Winit([4, plp["pweight_num"]])

                W = nn.Parameter(
                    torch.as_tensor(Wiv, dtype=torch.float32),
                    requires_grad=Wtrain,
                )
                channel.append(W)
            else:
                W = None

            layer, Ltrain = plp["layer"], plp["layer_train"]

            if layer == "PermutationEquivariant":
                Lpeq = plp["lpeq"]
                LWinit = plp["lweight_init"]
                LBinit = plp["lbias_init"]
                LGinit = plp["lgamma_init"]

                LW, LB, LG = [], [], []

                for idx, (dim, pop) in enumerate(Lpeq):
                    dim_before = self.diagdim if idx == 0 else Lpeq[idx - 1][0]

                    LWiv = LWinit([dim_before, dim]) if callable(LWinit) else LWinit
                    LBiv = LBinit([dim]) if callable(LBinit) else LBinit

                    LW_parameter = nn.Parameter(
                        torch.as_tensor(LWiv, dtype=torch.float32),
                        requires_grad=Ltrain,
                    )
                    LB_parameter = nn.Parameter(
                        torch.as_tensor(LBiv, dtype=torch.float32),
                        requires_grad=Ltrain,
                    )
                    channel.extend([LW_parameter, LB_parameter])
                    LW.append(LW_parameter)
                    LB.append(LB_parameter)

                    if pop is not None:
                        LGiv = LGinit([dim_before, dim]) if callable(LGinit) else LGinit
                        LG_parameter = nn.Parameter(
                            torch.as_tensor(LGiv, dtype=torch.float32),
                            requires_grad=Ltrain,
                        )
                        channel.append(LG_parameter)
                        LG.append(LG_parameter)
                    else:
                        LG.append(None)

                lvars = [LW, LB, LG]

            elif layer in ("Landscape", "BettiCurve", "Entropy"):
                LSinit = plp["lsample_init"]
                LSiv = LSinit if not callable(LSinit) else LSinit([plp["lsample_num"]])
                LS = nn.Parameter(
                    torch.as_tensor(LSiv, dtype=torch.float32),
                    requires_grad=Ltrain,
                )
                channel.append(LS)
                lvars = LS

            elif layer == "Image":
                LVinit = plp["lvariance_init"]
                LViv = LVinit if not callable(LVinit) else LVinit([1])
                LV = nn.Parameter(
                    torch.as_tensor(LViv, dtype=torch.float32),
                    requires_grad=Ltrain,
                )
                channel.append(LV)
                lvars = LV

            elif layer == "Exponential":
                LMinit, LVinit = plp["lmean_init"], plp["lvariance_init"]
                LMiv = LMinit if not callable(LMinit) else LMinit(
                    [self.diagdim, plp["lnum"]]
                )
                LViv = LVinit if not callable(LVinit) else LVinit(
                    [self.diagdim, plp["lnum"]]
                )

                LM = nn.Parameter(
                    torch.as_tensor(LMiv, dtype=torch.float32),
                    requires_grad=Ltrain,
                )
                LV = nn.Parameter(
                    torch.as_tensor(LViv, dtype=torch.float32),
                    requires_grad=Ltrain,
                )
                channel.extend([LM, LV])
                lvars = [LM, LV]

            elif layer == "Rational":
                LMinit = plp["lmean_init"]
                LVinit = plp["lvariance_init"]
                LAinit = plp["lalpha_init"]

                LMiv = LMinit if not callable(LMinit) else LMinit(
                    [self.diagdim, plp["lnum"]]
                )
                LViv = LVinit if not callable(LVinit) else LVinit(
                    [self.diagdim, plp["lnum"]]
                )
                LAiv = LAinit if not callable(LAinit) else LAinit([plp["lnum"]])

                LM = nn.Parameter(
                    torch.as_tensor(LMiv, dtype=torch.float32),
                    requires_grad=Ltrain,
                )
                LV = nn.Parameter(
                    torch.as_tensor(LViv, dtype=torch.float32),
                    requires_grad=Ltrain,
                )
                LA = nn.Parameter(
                    torch.as_tensor(LAiv, dtype=torch.float32),
                    requires_grad=Ltrain,
                )
                channel.extend([LM, LV, LA])
                lvars = [LM, LV, LA]

            elif layer == "RationalHat":
                LMinit, LRinit = plp["lmean_init"], plp["lr_init"]
                LMiv = LMinit if not callable(LMinit) else LMinit(
                    [self.diagdim, plp["lnum"]]
                )
                LRiv = LRinit if not callable(LRinit) else LRinit([1])

                LM = nn.Parameter(
                    torch.as_tensor(LMiv, dtype=torch.float32),
                    requires_grad=Ltrain,
                )
                LR = nn.Parameter(
                    torch.as_tensor(LRiv, dtype=torch.float32),
                    requires_grad=Ltrain,
                )
                channel.extend([LM, LR])
                lvars = [LM, LR]

            self._parameters_by_channel.append(channel)
            self.vars.append([W, lvars])

            final_model = plp["final_model"]
            if final_model != "identity" and isinstance(final_model, nn.Module):
                self._final_models[str(nf)] = final_model

    def compute_representations(self, diags, training=False):
        list_v = []

        for nf, plp in enumerate(self.perslay_parameters):
            diag = diags[nf]

            if not isinstance(diag, torch.Tensor):
                diag = torch.as_tensor(diag, dtype=torch.float32)

            device = next(self.parameters()).device
            diag = diag.to(device=device, dtype=torch.float32)

            dimension_diag = diag.shape[2]
            tensor_mask = diag[:, :, dimension_diag - 1]
            tensor_diag = diag[:, :, :dimension_diag - 1]

            W = self.vars[nf][0]

            if plp["pweight"] == "power":
                p = plp["pweight_power"]
                weight = W * torch.pow(
                    torch.abs(tensor_diag[:, :, 1:2] - tensor_diag[:, :, 0:1]),
                    p,
                )

            elif plp["pweight"] == "grid":
                indices = []
                for dim in range(dimension_diag - 1):
                    m, M = plp["pweight_bnds"][dim]
                    coords = tensor_diag[:, :, dim:dim + 1]
                    ids = W.shape[dim] * (coords - m) / (M - m)
                    indices.append(ids.long())

                indices = torch.cat(indices, dim=2)
                index_tuple = tuple(
                    indices[:, :, dim]
                    for dim in range(indices.shape[2])
                )
                weight = W[index_tuple].unsqueeze(-1)

            elif plp["pweight"] == "gmix":
                M = W[:2, :].unsqueeze(0).unsqueeze(0)
                V = W[2:, :].unsqueeze(0).unsqueeze(0)
                bc_inp = tensor_diag.unsqueeze(-1)
                weight = torch.sum(
                    torch.exp(
                        torch.sum(
                            -torch.square(bc_inp - M) * torch.square(V),
                            dim=2,
                        )
                    ),
                    dim=2,
                ).unsqueeze(-1)

            else:
                weight = None

            lvars = self.vars[nf][1]

            if plp["layer"] == "PermutationEquivariant":
                for idx, (dim, pop) in enumerate(plp["lpeq"]):
                    tensor_diag = permutation_equivariant_layer(
                        tensor_diag,
                        dim,
                        pop,
                        lvars[0][idx],
                        lvars[1][idx],
                        lvars[2][idx],
                    )
            elif plp["layer"] == "Landscape":
                tensor_diag = landscape_layer(tensor_diag, lvars)
            elif plp["layer"] == "BettiCurve":
                tensor_diag = betti_layer(tensor_diag, plp["theta"], lvars)
            elif plp["layer"] == "Entropy":
                tensor_diag = entropy_layer(tensor_diag, plp["theta"], lvars)
            elif plp["layer"] == "Image":
                tensor_diag = image_layer(
                    tensor_diag,
                    plp["image_size"],
                    plp["image_bnds"],
                    lvars,
                )
            elif plp["layer"] == "Exponential":
                tensor_diag = exponential_layer(
                    tensor_diag,
                    lvars[0],
                    lvars[1],
                )
            elif plp["layer"] == "Rational":
                tensor_diag = rational_layer(
                    tensor_diag,
                    lvars[0],
                    lvars[1],
                    lvars[2],
                )
            elif plp["layer"] == "RationalHat":
                tensor_diag = rational_hat_layer(
                    tensor_diag,
                    plp["q"],
                    lvars[0],
                    lvars[1],
                )

            output_dim = len(tensor_diag.shape) - 2

            if weight is not None:
                for _ in range(output_dim - 1):
                    weight = weight.unsqueeze(-1)
                tensor_diag = tensor_diag * weight

            for _ in range(output_dim):
                tensor_mask = tensor_mask.unsqueeze(-1)

            masked_layer = tensor_diag * tensor_mask

            if plp["perm_op"] == "topk" and output_dim == 1:
                masked_layer_t = masked_layer.transpose(1, 2)
                values, _ = torch.topk(masked_layer_t, k=plp["keep"], dim=2)
                vector = values.reshape(
                    -1,
                    plp["keep"] * tensor_diag.shape[2],
                )
            elif plp["perm_op"] == "sum":
                vector = torch.sum(masked_layer, dim=1)
            elif plp["perm_op"] == "max":
                vector = torch.max(masked_layer, dim=1).values
            elif plp["perm_op"] == "mean":
                vector = torch.mean(masked_layer, dim=1)

            if plp["final_model"] != "identity":
                vector = self._final_models[str(nf)](vector)

            list_v.append(vector)

        return torch.cat(list_v, dim=1)

    def forward(self, inputs, training=False):
        diags, feats = inputs[0], inputs[1]
        representations = self.compute_representations(diags, training)

        if not isinstance(feats, torch.Tensor):
            feats = torch.as_tensor(feats, dtype=representations.dtype)

        feats = feats.to(
            device=representations.device,
            dtype=representations.dtype,
        )
        concat_representations = torch.cat([representations, feats], dim=1)

        if self.rho != "identity":
            return self.rho(concat_representations)

        return concat_representations

    def call(self, inputs, training=False):
        return self.forward(inputs, training)