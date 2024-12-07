import json
from collections import Counter
from copy import copy
from pathlib import Path

import spacy
# noinspection PyUnresolvedReferences
from fastcoref import spacy_component

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe("fastcoref")


class GameSeq:
    K = 6

    def __init__(self, seq: str | list[tuple[int, int] | str]):
        """
        If constructing from a string, parse the string.
        If constructing from a list, build directly.
        """
        if type(seq) is str:
            # have to do the work to parse the string
            str_seq = list(seq.split("#"))
            seq = []
            for s in str_seq:
                t = s.split(",")
                if len(t) == 2:
                    seq.append((int(t[0]), int(t[1])))
                else:
                    seq.append(s)
        self.seq = tuple(seq)

    def __str__(self) -> str:
        """
        Convert a game sequence to string, with no spaces.
        """

        def np_to_string(np: tuple[int, int] | str) -> str:
            if type(np) == str:
                return np
            else:
                a, b = np
                return f"{a},{b}"

        str_seq = [np_to_string(np) for np in self.seq]
        return "#".join(str_seq)

    def __hash__(self) -> int:
        return hash(self.seq)

    def put_placeholders(self) -> list[any]:
        """
        For each non-unique noun phrase, either replace it with a PLACEHOLDER or leave it unchanged.
        """
        # ugh i think i have to copy here probably
        seqs = [copy(self)]
        for i in range(self.K):
            nxt = []
            for game_seq in seqs:
                if game_seq.seq[i] == "UNIQUE":
                    nxt.append(game_seq)
                else:
                    nxt.append(game_seq)
                    seq = list(game_seq.seq)
                    seq[i] = "PLACEHOLDER"
                    nxt.append(GameSeq(seq))
            seqs = nxt
        return seqs

    def resolve_placeholders(self, entities: list[tuple[int, int]]) -> list[any]:
        """
        For each PLACEHOLDER, replace it with an entity.
        """
        seqs = [copy(self)]
        for i in range(self.K):
            nxt = []
            for game_seq in seqs:
                if game_seq.seq[i] == "PLACEHOLDER":
                    for entity in entities:
                        seq = list(game_seq.seq)
                        seq[i] = entity
                        nxt.append(GameSeq(seq))
                else:
                    nxt.append(game_seq)
            seqs = nxt
        return seqs

    def score(self, resolved_seq):
        # start at 100 to eliminate bad game states
        score = 100
        for a, b in zip(self.seq, resolved_seq.seq):
            if a == b and a != "UNIQUE":
                score += 1
            elif a != b:
                score -= 10
        return score


def build_game(prefix_text: str, completions: list[str], game_file: Path):
    """
    Build the game.
    """
    prefix_doc = nlp(prefix_text)

    game_seq_counter = Counter()
    entities = []

    total_utility = 0

    for text in completions:
        doc = nlp(text)
        prefix_end = len(prefix_doc)

        cluster = {char_span: entity[0] for entity in doc._.coref_clusters for char_span in entity}
        if len(set(cluster.values())) > len(entities):
            entities = list(set(cluster.values()))

        game_seq = []
        realization = []
        for noun_chunk in doc.noun_chunks:
            char_span = (noun_chunk.start_char, noun_chunk.end_char)
            if prefix_end < noun_chunk.start:
                realization.append(noun_chunk)
                if char_span in cluster:
                    game_seq.append(cluster[char_span])
                else:
                    game_seq.append("UNIQUE")

        if len(game_seq) >= GameSeq.K:
            game_seq_counter[str(GameSeq(game_seq[:GameSeq.K]))] += 1

            total_utility += 100
            for noun_chunk in realization[:GameSeq.K]:
                if noun_chunk.start + 1 == noun_chunk.end and noun_chunk[0].pos_ == "PRON":
                    total_utility += 1

    decision_problem_pl1 = [{
        "id": "observe_entities1",
        "type": "observation",
        "signals": list(game_seq_counter.keys()),
        "parent_edge": None,
    }]
    replaced_seqs = set()
    for game_seq_str in game_seq_counter:
        game_seq = GameSeq(game_seq_str)
        decision_problem_pl1.append({
            "id": str(game_seq),
            "type": "decision",
            "actions": list(map(str, game_seq.put_placeholders())),
            "parent_edge": (
                "observe_entities1",
                str(game_seq)
            ),
            "parent_sequence": None,
        })
        replaced_seqs |= set(game_seq.put_placeholders())

    decision_problem_pl2 = [{
        "id": "observe_entities2",
        "type": "observation",
        "signals": [str(replaced_seq) for replaced_seq in replaced_seqs],
        "parent_edge": None,
    }]
    for replaced_seq in replaced_seqs:
        decision_problem_pl2.append({
            "id": str(replaced_seq),
            "type": "decision",
            "actions": list(map(str, replaced_seq.resolve_placeholders(entities))),
            "parent_edge": (
                "observe_entities2",
                str(replaced_seq)
            ),
            "parent_sequence": None,
        })

    utility_pl1 = []
    for game_seq_str, count in game_seq_counter.items():
        game_seq = GameSeq(game_seq_str)
        for replaced_seq in game_seq.put_placeholders():
            for resolved_seq in replaced_seq.resolve_placeholders(entities):
                utility_pl1.append({
                    "sequence_pl1": (
                        str(game_seq),
                        str(replaced_seq)
                    ),
                    "sequence_pl2": (
                        str(replaced_seq),
                        str(resolved_seq)
                    ),
                    "value": count * game_seq.score(resolved_seq),
                })

    with open(game_file, "w") as f:
        json.dump({
            "decision_problem_pl1": decision_problem_pl1,
            "decision_problem_pl2": decision_problem_pl2,
            "utility_pl1": utility_pl1,
        }, f)

    print(f"terminal node count: {len(utility_pl1)}")
    print(f"distinct entity sequences: {len(game_seq_counter.keys())}")
    print(f"distribution of the tail: {game_seq_counter.most_common(5)}")
    print(f"total utility: {total_utility}")
