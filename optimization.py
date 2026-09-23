"""Parameter utilities for the MosaicADAPT-QAOA ansatz."""


def pack_parameters(layers):
    """Flatten parameters as [all gammas, then all betas]."""
    gamma_parameters = [
        float(layer["gamma"])
        for layer in layers
    ]

    beta_parameters = [
        float(beta)
        for layer in layers
        for beta in layer["betas"]
    ]

    return gamma_parameters + beta_parameters


def copy_layer_structure(layers):
    """Copy layer containers without deep-copying CUDA-Q operators.

    CUDA-Q operator objects are backed by C++ objects and do not need to
    be copied during optimization. Only numerical gamma and beta values
    are changed.
    """
    copied_layers = []

    for layer in layers:
        copied_layer = dict(layer)
        copied_layer["operators"] = list(layer["operators"])
        copied_layer["betas"] = [
            float(beta)
            for beta in layer["betas"]
        ]
        copied_layer["gamma"] = float(layer["gamma"])
        copied_layers.append(copied_layer)

    return copied_layers


def unpack_parameters(parameters, template_layers):
    """Write [all gammas, then all betas] into a layer structure."""
    parameters = list(parameters)
    layers = copy_layer_structure(template_layers)

    number_of_layers = len(layers)
    number_of_betas = sum(
        len(layer["betas"])
        for layer in layers
    )
    expected_parameter_count = (
        number_of_layers + number_of_betas
    )

    if len(parameters) != expected_parameter_count:
        raise ValueError(
            "Parameter vector has incorrect size: "
            f"expected {expected_parameter_count}, "
            f"received {len(parameters)}."
        )

    for layer_index in range(number_of_layers):
        layers[layer_index]["gamma"] = float(
            parameters[layer_index]
        )

    parameter_index = number_of_layers

    for layer in layers:
        number_of_layer_betas = len(layer["betas"])

        layer["betas"] = [
            float(parameters[parameter_index + offset])
            for offset in range(number_of_layer_betas)
        ]

        parameter_index += number_of_layer_betas

    return layers
