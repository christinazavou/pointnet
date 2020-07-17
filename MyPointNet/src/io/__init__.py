import argparse
from configparser import ConfigParser
import sys


def parse_configuration(argv=None):
    # Do argv default this way, as doing it in the functional
    # declaration sets it at compile time.
    if argv is None:
        argv = sys.argv

    # Parse any conf_file specification
    # We make this parser with add_help=False so that
    # it doesn't parse -h and print help.
    conf_parser = argparse.ArgumentParser(
        description=__doc__, # printed with -h/--help
        # Don't mess with format of description
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # Turn off help, so we print all options in response to -h
        add_help=False
        )
    conf_parser.add_argument("-c", "--conf_file", help="Specify config file", metavar="FILE")
    args, remaining_argv = conf_parser.parse_known_args()
    print("args:", args)
    print("remaining_argv:", remaining_argv)

    defaults = {}

    if args.conf_file:
        config = ConfigParser()
        config.read([args.conf_file])
        defaults.update(dict(config.items("Defaults")))

    # Parse rest of arguments
    # Don't suppress add_help here so it will handle -h
    parser = argparse.ArgumentParser(
        # Inherit options from config_parser
        parents=[conf_parser]
        )
    parser.set_defaults(**defaults)
    for arg in defaults:
        parser.add_argument("--"+arg)

    args = parser.parse_args(remaining_argv)
    print(args)
    return args

