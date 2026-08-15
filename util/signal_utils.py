"""Report summary statistics for auxiliary feedback signals."""


def log_signal_density(name, mat, user_count, item_count):
    """Print the nonzero count and density of a signal matrix."""
    density = 100.0 * mat.nnz / (user_count * item_count)
    print(f"    [signal] {name}: nnz={mat.nnz:,} density={density:.2f}%")
