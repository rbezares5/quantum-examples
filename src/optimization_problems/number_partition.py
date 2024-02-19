"""Number Partition problem solver script.

Minimize (a_1*x_1 + a_2*x_2 + ...)^2

where:
    a is the list of numbers
    x is the decision variable (1 if belongs to A, -1 if belongs to B)
"""

import dimod
import dwave.system
import neal
import numpy as np
import pyqubo

# Online solver parameters
ENDPOINT = "https://eu-central-1.cloud.dwavesys.com/sapi/v2/"
SOLVER = "Advantage_system5.4"
TOKEN = ""  # input your token or configure solver with "dwave config create"



def print_sample(data: np.ndarray, sol: dimod.SampleSet) -> None:
    """Display the result in a comprehensive way.

    This function takes the initial data and the solution obtained as parameters.
    """
    a = []
    b = []

    print("Best solution found", end="")  # noqa: T201
    if sol.energy != 0:
        print(" (energy not equal to 0)", end="")  # noqa: T201
    print(":")  # noqa: T201

    for i in range(len(data)):
        if sol.array("x", (i)) == 1:
            a.append(data[i])

        else:
            b.append(data[i])

    print(f"A = {a} (sum = {np.sum(a)})\nB = {b} (sum = {np.sum(b)})")  # noqa: T201


def number_partition(data: str="dummy", sampler: str="sim") -> None:
    """Problem solver implementation function."""
    # Generate problem data or take sample dummy data
    rng = np.random.default_rng()   # First create instance of np.Generator class
    a = rng.integers(low=1, high=5, size=10) if data == "random" else [2, 3, 4, 3, 2]
    print(f"Starting data:\n{a}\n")  # noqa: T201

    # Decision variables
    s = pyqubo.Array.create("x", shape=(len(a)), vartype="SPIN")

    # Constraints
    # N/A

    # Objective Function -> (a_1*x_1 + a_2*x_2 + ...)^2
    objective_function = s.dot(a) ** 2  # H

    # Compile and create binary-quadratic-maximization model
    model = objective_function.compile()
    bqm = model.to_bqm()

    # Get the sampler, since this is testing, the simulated annealer is fine
    if sampler == "online":
        # Use the real online annealer

        sampler = dwave.system.composites.EmbeddingComposite(
            dwave.system.samplers.DWaveSampler(
                endpoint=ENDPOINT, token=TOKEN, solver=SOLVER,
            ),
        )
    else:
        sampler = neal.SimulatedAnnealingSampler()

    # Get response
    response = sampler.sample(
        bqm, num_reads=5, chain_strength=70,
    )  # for this problem a high chian strength yields better results
    print(response)  # noqa: T201

    decoded_samples = model.decode_sampleset(response)
    best_sample = min(decoded_samples, key=lambda x: x.energy)

    # if energy>0, not optimal solution (probably doesn't exist for the given data)
    print_sample(a, best_sample)


if __name__ == "__main__":
    number_partition(data="sim", sampler="online")
