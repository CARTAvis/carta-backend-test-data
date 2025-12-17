import argparse

def generate_tile_data(width, height, mode, output_file, start=0.0, step=1.0, value=0.0, direction="horizontal"):
    """
    Generates 2D tile data with single or gradient progression and writes it to a text file.

    Args:
        width (int): Number of columns.
        height (int): Number of rows.
        mode (str): 'single' or 'gradient'.
        output_file (str): Output filename.
        start (float): Starting value.
        step (float): Increment per unit.
        value (float): Single value if mode='single'.
        direction (str): 'horizontal', 'vertical', or 'diagonal' for gradients.
    """
    with open(output_file, "w") as f:
        for y in range(height):
            row = []
            for x in range(width):
                if mode == "single":
                    val = value
                elif mode == "gradient":
                    if direction == "horizontal":
                        val = start + x * step
                    elif direction == "vertical":
                        val = start + y * step
                    elif direction == "diagonal":
                        val = start + (x + y) * step
                    else:
                        raise ValueError("Invalid direction for gradient")
                else:
                    raise ValueError("Mode must be 'single' or 'gradient'")
                row.append(f"{val:.2f}")
            f.write(" ".join(row) + "\n")

    print(f"Tile data written to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 2D tile data with gradient or single value.")
    parser.add_argument("--width", type=int, required=True, help="Number of columns")
    parser.add_argument("--height", type=int, required=True, help="Number of rows")
    parser.add_argument("--mode", choices=["single", "gradient"], required=True, help="Generation mode")
    parser.add_argument("--output", type=str, default="tile_data.txt", help="Output file name")
    parser.add_argument("--start", type=float, default=0.0, help="Start value for gradient")
    parser.add_argument("--step", type=float, default=1.0, help="Increment per step for gradient")
    parser.add_argument("--value", type=float, default=0.0, help="Value for single mode")
    parser.add_argument("--direction", choices=["horizontal", "vertical", "diagonal"], default="horizontal",
                        help="Direction of gradient progression")

    args = parser.parse_args()

    generate_tile_data(
        args.width,
        args.height,
        args.mode,
        args.output,
        args.start,
        args.step,
        args.value,
        args.direction
    )
