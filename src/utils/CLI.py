import argparse
def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dir",
        dest="directory",
        default="./test_data",
        type=str,
        help="Directory path to load song from",
    )

    parser.add_argument(
        "-s",
        "--song", 
        type=str, 
        default="The Chain - Fleetwood Mac (balanced).wav",
        dest="song",
        help="Song name/destination"
    )

    parser.add_argument(
        "-q",
        "--quality", 
        type=str, 
        choices=["low", "medium", "high"],
        default="medium",
        dest="quality",
        help="Video / render quality"
    )

    parser.add_argument(
        "-o",
        "--opacity", 
        type=float, 
        default=0.7,
        dest="opacity",
        help="bar opacity"
    )

    parser.add_argument(
        "-x",
        "--translate-x", 
        type=float, 
        default=-7,
        dest="translate_x",
        help="Bar x coordinate translation. Applied to all"
    )

    parser.add_argument(
        "-y",
        "--translate-y", 
        type=float, 
        default=0,
        dest="translate_y",
        help="Bar y coordinate translation. Applied to all"
    )

    parser.add_argument(
        "-z",
        "--translate-z", 
        type=float, 
        default=1,
        dest="translate_z",
        help="Bar z coordinate translation. Applied to all"
    )

    args = parser.parse_args()
    return args