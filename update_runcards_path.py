import argparse
import pathlib

import yaml

here = pathlib.Path(__file__).parent.absolute()


def load_base_runcard(base_runcard_path):
    with open(base_runcard_path, "r", encoding="utf-8") as f:
        base_runcard = yaml.safe_load(f)
    return base_runcard


def update_paths(loaded_runcard, destination):
    # path to common data
    loaded_runcard["data_path"] = here.joinpath("commondata").as_posix()
    # path to theory tables, default same as data path
    loaded_runcard["theory_path"] = here.joinpath("theory").as_posix()
    # absolute path where results are stored
    loaded_runcard["result_path"] = destination.joinpath("results").as_posix()
    return loaded_runcard


def dump_runcard(card, file_name, destination):
    with open(destination / f"{file_name}.yaml", "w", encoding="utf-8") as f:
        yaml.dump(card, f, sort_keys=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Setup smefit runcards updating the paths."
    )
    parser.add_argument(
        "base_runcard", type=pathlib.Path, help="path to base runcards to update"
    )
    parser.add_argument(
        "-d",
        "--destination",
        type=pathlib.Path,
        default=here,
        help="path to smefit runcards folder",
    )
    args = parser.parse_args()

    loaded_runcard = load_base_runcard(args.base_runcard.absolute())
    loaded_runcard = update_paths(loaded_runcard, args.destination.absolute().parent)
    dump_runcard(loaded_runcard, args.base_runcard.stem, args.destination.absolute())
