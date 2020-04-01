import argparse
import os
import sys
import subprocess
import time

from .yamato_utils import (
    get_base_path,
    run_standalone_build,
    init_venv,
    override_config_file,
    checkout_csharp_version,
    undo_git_checkout,
)

import torch


def test_pytorch():
    # Copied from https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-nn

    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 64, 1000, 100, 10

    # Create random Tensors to hold inputs and outputs
    x = torch.randn(N, D_in)
    y = torch.randn(N, D_out)

    # Use the nn package to define our model as a sequence of layers. nn.Sequential
    # is a Module which contains other Modules, and applies them in sequence to
    # produce its output. Each Linear Module computes output from input using a
    # linear function, and holds internal Tensors for its weight and bias.
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H), torch.nn.ReLU(), torch.nn.Linear(H, D_out)
    )

    # The nn package also contains definitions of popular loss functions; in this
    # case we will use Mean Squared Error (MSE) as our loss function.
    loss_fn = torch.nn.MSELoss(reduction="sum")

    learning_rate = 1e-4
    for t in range(5):
        # Forward pass: compute predicted y by passing x to the model. Module objects
        # override the __call__ operator so you can call them like functions. When
        # doing so you pass a Tensor of input data to the Module and it produces
        # a Tensor of output data.
        y_pred = model(x)

        # Compute and print loss. We pass Tensors containing the predicted and true
        # values of y, and the loss function returns a Tensor containing the
        # loss.
        loss = loss_fn(y_pred, y)
        if t % 100 == 99:
            print(t, loss.item())

        # Zero the gradients before running the backward pass.
        model.zero_grad()

        # Backward pass: compute gradient of the loss with respect to all the learnable
        # parameters of the model. Internally, the parameters of each Module are stored
        # in Tensors with requires_grad=True, so this call will compute gradients for
        # all learnable parameters in the model.
        loss.backward()

        # Update the weights using gradient descent. Each parameter is a Tensor, so
        # we can access its gradients like we did before.
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad
    print("PYTORCH DIDN'T BREAK!!!")


def run_training(python_version, csharp_version):
    latest = "latest"
    run_id = int(time.time() * 1000.0)
    print(
        f"Running training with python={python_version or latest} and c#={csharp_version or latest}"
    )
    nn_file_expected = f"./models/{run_id}/3DBall.nn"
    if os.path.exists(nn_file_expected):
        # Should never happen - make sure nothing leftover from an old test.
        print("Artifacts from previous build found!")
        sys.exit(1)

    base_path = get_base_path()
    print(f"Running in base path {base_path}")

    # Only build the standalone player if we're overriding the C# version
    # Otherwise we'll use the one built earlier in the pipeline.
    if csharp_version is not None:
        # We can't rely on the old C# code recognizing the commandline argument to set the output
        # So rename testPlayer (containing the most recent build) to something else temporarily
        full_player_path = os.path.join("Project", "testPlayer.app")
        temp_player_path = os.path.join("Project", "temp_testPlayer.app")
        final_player_path = os.path.join("Project", f"testPlayer_{csharp_version}.app")

        os.rename(full_player_path, temp_player_path)

        checkout_csharp_version(csharp_version)
        build_returncode = run_standalone_build(base_path)

        if build_returncode != 0:
            print("Standalone build FAILED!")
            sys.exit(build_returncode)

        # Now rename the newly-built executable, and restore the old one
        os.rename(full_player_path, final_player_path)
        os.rename(temp_player_path, full_player_path)
        standalone_player_path = f"testPlayer_{csharp_version}"
    else:
        standalone_player_path = "testPlayer"

    venv_path = init_venv(python_version)

    # Copy the default training config but override the max_steps parameter,
    # and reduce the batch_size and buffer_size enough to ensure an update step happens.
    override_config_file(
        "config/trainer_config.yaml",
        "override.yaml",
        max_steps=100,
        batch_size=10,
        buffer_size=10,
    )

    mla_learn_cmd = (
        f"mlagents-learn override.yaml --train --env=Project/{standalone_player_path} "
        f"--run-id={run_id} --no-graphics --env-args -logFile -"
    )  # noqa
    res = subprocess.run(
        f"source {venv_path}/bin/activate; {mla_learn_cmd}", shell=True
    )

    if res.returncode != 0 or not os.path.exists(nn_file_expected):
        print("mlagents-learn run FAILED!")
        sys.exit(1)

    print("mlagents-learn run SUCCEEDED!")
    sys.exit(0)


def main():
    test_pytorch()

    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=None)
    parser.add_argument("--csharp", default=None)
    args = parser.parse_args()

    try:
        run_training(args.python, args.csharp)
    finally:
        # Cleanup - this gets executed even if we hit sys.exit()
        undo_git_checkout()


if __name__ == "__main__":
    main()
