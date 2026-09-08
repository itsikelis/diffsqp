import torch

from diffsqp.types import SqpSolution


def save_solution(solution: SqpSolution, filepath: str, x_des=None) -> None:
    """Saves only the x and u tensors for warm starting."""
    # Best practice: Move tensors to the CPU before saving.
    # This ensures you won't get CUDA errors if you try to load them on
    # a machine with a different GPU setup or no GPU at all.
    batch_size = solution.x.shape[0]
    warmstart_data = {
        "x": solution.x.detach().cpu(),
        "u": solution.u.detach().cpu(),
        "x_des": x_des.detach().cpu().repeat(batch_size, 1),
    }
    torch.save(warmstart_data, filepath)
    print(f"Trajectory saved to {filepath}.")


def load_solution(
    filepath: str, device: torch.device = torch.device("cpu")
) -> tuple[torch.Tensor, torch.Tensor]:
    """Loads x and u tensors and sends them to the target device."""
    print(f"Loading trajectory from {filepath}.")
    data = torch.load(filepath, map_location=torch.device(device), weights_only=True)
    x = data["x"]
    u = data["u"]
    return x, u
