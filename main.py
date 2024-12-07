import argparse
import sys
from pathlib import Path

from generate import generate
from build_game import build_game
from stub import solve

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='coreference_game',
        # TODO: add reference to my paper
        # TODO: update the default value of `n`
        # TODO: check that this whole thing is accurate
        description=
        """
        A model of coreference resolution as a two-player imperfect information game, as described in TODO.
        """,
        usage=
        """
        There are several different commands this program provides. 
        
        Generate sentence completions from `prefix_file` and store the sentence completions as files
        in `completion_dir/', with the names `1.txt`, `2.txt`, ... `n.txt`. `n` defaults to 200.
            main.py generate --prefix <prefix_file> --completion_dir <completion_dir/> [-n]
            
        Build and save the game. This runs the Stanford coreference resolver to determine entities and game states.
        It also reports the number of terminal nodes in the game, the number of distinct entity sequences,
        the distribution of the five highest-frequency entity sequences, and the total utility that would be achieved
        using the generated text's pronoun assignments.
            main.py build_game --prefix <prefix_file> --completion_dir <completion_dir/> --game_file <game_file>
        
        Solve the game and plot its utility over iterations, as well as save the sequence-form strategy.
            main.py solve --game_file <game_file> --plot_dir <plot_dir>
            
        So typically, you call generate, then build, then solve.
        """
    )
    parser.add_argument('mode', choices=['generate', 'build_game', 'solve'])
    parser.add_argument('--prefix', type=Path)
    parser.add_argument('--completion_dir', type=Path)
    parser.add_argument('--game_file', type=Path)
    parser.add_argument('--plot_dir', type=Path)
    parser.add_argument('-n', type=int, default=100)
    args = parser.parse_args()

    if args.mode == 'generate':
        if args.prefix is None:
            print('No prefix specified', file=sys.stderr)
            parser.print_help()
            sys.exit(1)
        if args.completion_dir is None:
            print('No completion directory specified', file=sys.stderr)
            parser.print_help()
            sys.exit(1)
        with open(args.prefix) as prefix_file:
            generate(prefix_file.read(), args.completion_dir, args.n)
    elif args.mode == 'build_game':
        if args.prefix is None:
            print('No prefix specified', file=sys.stderr)
            parser.print_help()
            sys.exit(1)
        if args.completion_dir is None:
            print('No completion directory specified', file=sys.stderr)
            parser.print_help()
            sys.exit(1)
        if args.game_file is None:
            print('No game file specified', file=sys.stderr)
            parser.print_help()
            sys.exit(1)
        with open(args.prefix) as prefix_file:
            prefix_text = prefix_file.read()
            completions = []
            for completion_file in args.completion_dir.iterdir():
                with open(completion_file) as f:
                    completions.append(f.read())
            build_game(prefix_text, completions, args.game_file)
    elif args.mode == 'solve':
        if args.game_file is None:
            print('No game file specified', file=sys.stderr)
            parser.print_help()
            sys.exit(1)
        if args.plot_dir is None:
            print('No plot directory specified', file=sys.stderr)
            parser.print_help()
            sys.exit(1)
        solve(args.game_file, args.plot_dir)
    else:
        print('Unknown mode', file=sys.stderr)
        parser.print_help()
        sys.exit(1)
