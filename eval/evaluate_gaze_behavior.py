import argparse
import json



def evaluate(manifest):

    return {

        "metric":
        "gaze consistency",

        "status":
        "not implemented"

    }



def main():

    parser=argparse.ArgumentParser()


    parser.add_argument(
        "--manifest"
    )


    parser.add_argument(
        "--output"
    )


    args=parser.parse_args()


    result=evaluate(
        args.manifest
    )


    with open(
        args.output,
        "w"
    ) as f:

        json.dump(
            result,
            f,
            indent=4
        )


if __name__=="__main__":
    main()