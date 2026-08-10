import argparse
import json
from pathlib import Path


def evaluate_pose(manifest):
    """
    Pose standardization evaluation skeleton.

    Future implementation:
    Compare generated face pose with canonical pose.
    """

    result = {
        "metric": "pose error",
        "status": "not implemented",
        "input_manifest": manifest
    }

    return result



def main():

    parser = argparse.ArgumentParser(
        description="Evaluate pose standardization"
    )

    parser.add_argument(
        "--manifest",
        required=True,
        help="Input evaluation manifest"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output json file"
    )


    args = parser.parse_args()


    result = evaluate_pose(
        args.manifest
    )


    output_path = Path(
        args.output
    )


    output_path.parent.mkdir(
        exist_ok=True,
        parents=True
    )


    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=4
        )


if __name__ == "__main__":
    main()