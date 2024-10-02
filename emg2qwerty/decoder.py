# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc

import hashlib
import math
from collections.abc import Iterator
from dataclasses import dataclass, field, InitVar
from typing import Any, ClassVar

import kenlm
import numpy as np

from emg2qwerty.charset import CharacterSet, charset
from emg2qwerty.data import LabelData


def logsumexp(*xs: float) -> float:
    """Stable log-sum-exp to sum probabilities in log-space.
    Ref for example http://gregorygundersen.com/blog/2020/02/09/log-sum-exp/.

    We could use `scipy.special.logsumexp`, but it's slower owing to implicit
    `numpy.ndarray` conversion."""
    x_max = max(xs)
    if x_max == -np.inf:
        return -float(np.inf)
    return x_max + math.log(sum(math.exp(x - x_max) for x in xs))


@dataclass
class Decoder(abc.ABC):
    """Base class for a stateful decoder that takes in emissions and returns
    decoded label sequences."""

    _charset: CharacterSet = field(default_factory=charset)

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset decoder state."""
        raise NotImplementedError

    @abc.abstractmethod
    def decode(
        self,
        emissions: np.ndarray,
        timestamps: np.ndarray,
        finish: bool = False,
    ) -> LabelData:
        """Online decoding API that updates decoder state and returns the
        decoded sequence thus far.

        Args:
            emissions (`np.ndarray`): Emission probability matrix of shape
                (T, num_classes).
            timestamps (`np.ndarray`): Timestamps corresponding to emissions
                of shape (T, ).
        Return:
            A `LabelData` instance with the decoding thus far and their
                corresponding onset timestamps.
        """
        raise NotImplementedError

    def decode_batch(
        self,
        emissions: np.ndarray,
        emission_lengths: np.ndarray,
    ) -> list[LabelData]:
        """Offline decoding API that operates over a batch of emission logits.

        This simply loops over each batch element and calls `decode` in sequence.
        Override if a more efficient implementation is possible for the specific
        decoding algorithm.

        Args:
            emissions (`np.ndarray`): A batch of emission probability matrices
                of shape (T, N, num_classes).
            emission_lengths: An array of size N with the valid temporal lengths
                of each emission matrix in the batch after removal of padding.
        Return:
            A list of `LabelData` instances, one per batch item.
        """
        assert emissions.ndim == 3  # (T, N, num_classes)
        assert emission_lengths.ndim == 1
        N = emissions.shape[1]

        decodings = []
        for i in range(N):
            # Unpad emission matrix (T, N, num_classes) for batch entry and decode
            self.reset()
            decodings.append(
                self.decode(
                    emissions=emissions[: emission_lengths[i], i],
                    timestamps=np.arange(emission_lengths[i]),
                )
            )

        return decodings


@dataclass
class CTCGreedyDecoder(Decoder):
    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.decoding: list[int] = []
        self.timestamps: list[Any] = []
        self.prev_label = self._charset.null_class

    def decode(
        self,
        emissions: np.ndarray,
        timestamps: np.ndarray,
        finish: bool = False,
    ) -> LabelData:
        assert emissions.ndim == 2  # (T, num_classes)
        assert emissions.shape[1] == self._charset.num_classes
        assert len(emissions) == len(timestamps)

        for label, timestamp in zip(emissions.argmax(-1), timestamps):
            if label != self._charset.null_class and label != self.prev_label:
                self.decoding.append(label)
                self.timestamps.append(timestamp)
            self.prev_label = label

        return LabelData.from_labels(
            labels=self.decoding,
            timestamps=self.timestamps,
            _charset=self._charset,
        )


@dataclass
class TrieNode:
    """Prefix trie to maintain the decoding paths of a beam.

    We keep track of parent pointers to backtrack and apply deletes.
    We don't maintain child pointers so that when a beam is pruned after each
    timestep, the refcounts of nodes not in the path of higher probability
    decodings are automatically garbage collected. This places an implicit
    upper-bound on the number of leaves in the trie to the size of the beam.

    Additionally, not maintaining a dict of child nodes (and therefore allowing
    for duplicate children) allows accurate tracking of onset timestamps.
    As an example, consider the following two paths leading to the same
    decoding 'ab':

        Path 1:     a b b b b b b
        Path 2:     a a a a a a b
        Timestamps: 0 1 2 3 4 5 6

    The decoding trie in both cases encodes the same prefix 'ab' but differ in
    their timestamps - for token 'b', path 1 has a much earlier onset
    timestamp (t=1) compared to path 2 (t=6). There are three scenarios to
    consider:

        Scenario 1:
            Path 1 has a much higher probability than path 2 which gets kicked
            out of the beam mid-way (say at t=3). Path 2 doesn't get a
            chance to reach the token 'b' and we get the correct onset
            timestamp of 'b' (t=1) corresponding to path 1.
        Scenario 2:
            Inverse of scenario 1 - path 2 has a higher probability and path 1
            gets kicked out of the beam mid-way (say at t=3). The correct
            onset timestamp for 'b' should be t=6 corresponding to path 2.
            But unlike scenario 1, both paths reach the token 'b', and path 1
            gets there sooner. A unique prefix trie with children cache would
            mean that despite path 1 getting dropped out of the beam, token 'b'
            with the incorrect timestamp of t=1 will be returned for path 2.
            Allowing for duplicate children lets path 2 create a new child node
            for token 'b' with the correct onset timestamp t=6.
        Scenario 3:
            Both the paths have high enough probability to remain in the beam.
            Here, we need to balance between the onset timestamp being too
            early but belonging to lower prob beam entries vs belonging to high
            prob beams but being too late. Heuristically, we iterate the t-1
            beam entries from highest to lowest prob while extending the beam
            entries and this seems to work well in practice.
    """

    value: Any
    parent: TrieNode | None = None

    def child(self, value: Any) -> TrieNode:
        return self.__class__(value=value, parent=self)

    @property
    def values(self) -> Iterator[Any]:
        """Sequence of values on the path from root to this node."""
        # Iterate to avoid hitting max recursion depth
        values = []
        node: TrieNode | None = self
        while node is not None:
            values.append(node.value)
            node = node.parent
        return reversed(values)

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def depth(self) -> int:
        return 0 if self.parent is None else (self.parent.depth + 1)

    def __str__(self) -> str:
        parent = self.parent.value if self.parent is not None else None
        return f"TrieNode: value={self.value} parent={parent}"


@dataclass
class BeamState:
    """State corresponding to a beam entry (a beam is a collection of
    `BeamState` instances). Encompasses a prefix corresponding to decoding
    path along with its probability, onset timestamps, and language model
    states/scores.

    The label prefix and LM state-score pairs are kept track via two separate
    tries as they differ in deletion handing. The probability of the label
    prefix is the sum of the probability of the prefix ending in blank label
    and in non-blank label.

    Args:
        label_node (TrieNode): Reference to the label trie node for this state.
        lm_node (TrieNode): Reference to the LM trie node for this state.
            (default: ``None``)
        p_b (float): Probability (in log-space) of ending in blank.
            We maintain separate probabilities for a decoding ending in blank
            and non-blank to be able to differentiate between repeated tokens
            (separate by a blank) and repeated instances of a single token
            collapsed together (default: ``-np.inf``)
        p_nb (float): Probability (in log-space) of ending in non-blank.
            (default: ``-np.inf``)
        _hash (hashlib._Hash): `hashlib._Hash` object corresponding to this
            state's decoding for efficiently keying the decoded prefix into
            a dict. (default: ``None``)
    """

    label_node: TrieNode
    lm_node: TrieNode | None = None
    p_b: float = -np.inf
    p_nb: float = -np.inf
    _hash: InitVar[hashlib._Hash | None] = None

    def __post_init__(self, _hash: hashlib._Hash | None) -> None:
        # Hash object for efficiently keying the decoded prefix into a dict
        # independent of the length of the decoding / number of timesteps.
        if _hash is None:
            self.hash_ = hashlib.sha256()
            self.hash_.update(bytes(self.decoding))
        else:
            self.hash_ = _hash

    @classmethod
    def init(cls, blank_label: int, lm: kenlm.Model | None = None) -> BeamState:
        """Initialize a new BeamState with empty sequence (CTC blank label),
        probability of 1 for ending in blank and 0 for non-blank.

        The label trie is initialized with the blank label at the root
        and the LM trie with KenLM state corresponding to `<s>`."""

        # Label trie root with blank label and timestamp of 0
        label_node = TrieNode(value=(blank_label, 0))

        # LM trie root with KenLM state corresponding to `<s>` and 0 score
        lm_node = None
        if lm is not None:
            lm_state = kenlm.State()
            lm.BeginSentenceWrite(lm_state)
            lm_node = TrieNode(value=(lm_state, 0.0))

        return cls(
            label_node=label_node,
            lm_node=lm_node,
            p_b=0.0,  # Prob 1 (in log-space) for ending in blank
            p_nb=-np.inf,  # Prob 0 (in log-space) for ending in non-blank
        )

    @property
    def p_total(self) -> float:
        """Total probability (in log-space) of the decoding path leading to
        this state."""
        return logsumexp(self.p_b, self.p_nb)

    @property
    def label(self) -> int:
        """Last label corresponding to this decoding state."""
        return int(self.label_node.value[0])

    @property
    def timestamp(self) -> Any:
        """Last timestamp corresponding to this state."""
        return self.label_node.value[1]

    @property
    def decoding(self) -> list[int]:
        """Sequence of decoded labels in the path leading to this beam state,
        ignoring the blank label at the trie root."""
        return [value[0] for value in self.label_node.values][1:]

    @property
    def timestamps(self) -> list[Any]:
        """Sequence of onset timestamps corresponding to the decoded labels
        in the path leading to this beam state."""
        return [value[1] for value in self.label_node.values][1:]

    @property
    def lm_state(self) -> kenlm.State:
        """LM state corresponding to this beam state."""
        if self.lm_node is None:
            raise RuntimeError("Did you forget to call `init()` with lm?")
        return self.lm_node.value[0]

    @property
    def lm_states(self) -> list[kenlm.State]:
        """Sequence of LM states in the path leading to this beam state."""
        if self.lm_node is None:
            raise RuntimeError("Did you forget to call `init()` with lm?")
        return [value[0] for value in self.lm_node.values]

    @property
    def lm_score(self) -> float:
        """LM score corresponding to this beam node."""
        if self.lm_node is None:
            raise RuntimeError("Did you forget to call `init()` with lm?")
        return float(self.lm_node.value[1])

    @property
    def lm_scores(self) -> list[kenlm.State]:
        """Sequence of LM scores in the path leading to this beam state."""
        if self.lm_node is None:
            raise RuntimeError("Did you forget to call `init()` with lm?")
        return [value[1] for value in self.lm_node.values]

    def hash(self, next_label: int | None = None) -> hashlib._Hash:
        """`hashlib._Hash` object of the sequence of decoded labels in the path
        leading to this beam state for efficiently keying into a dict.

        If `next_label` is not None, the return hash object corresponds to
        extending the current decoding with `next_label`."""
        if next_label is None:
            return self.hash_

        _hash = self.hash_.copy()
        _hash.update(bytes([next_label]))
        return _hash

    def __str__(self) -> str:
        o = (
            f"BeamState: label={self.label}"
            f" len(decoding)={len(self.decoding)}"
            f" p_b={self.p_b} p_nb={self.p_nb} p_total={self.p_total}"
        )
        if self.lm_node is not None:
            o += (
                f" len(lm_states)={len(self.lm_states)}"
                f" lm_score={self.lm_score}"
                f" sum(lm_scores)={sum(self.lm_scores)}"
            )
        return o


@dataclass
class CTCBeamDecoder(Decoder):
    """CTC beam search lexicon-free decoder with a KenLM n-gram language model
    (modified Kneser-Ney) that also handles delete scenarios.

    Ref https://distill.pub/2017/ctc/ for an explanation of the standard
    CTC beam-search decoding algorithm. The implementation is largely inspired
    by github.com/facebookresearch/flashlight (LexiconFreeDecoder.cpp).

    The output decoding can contain delete labels as is (uncorrected), but
    care is taken to update LM states taking deletion into account.
    For example, if the best sequence of tokens is `c z ⌫ a t', the output
    decoding will be just that. But the LM score will be that of 'c a t',
    i.e., P(c | <s>) * P(a | <s> c) * P(t | <s> c a).
    This is ensured by maintaining two separate tries - one for the decoded
    label sequence and the other to keep track of LM states/scores. They differ
    in their updates only on encountering deletes, but are otherwise the same.

    LM is applied for a contiguous sequence of tokens that are in the LM
    vocabulary, and anything outside is treated as an out-of-vocabulary (OOV)
    token and given a baseline LM score. For instance, given a sequence
    'don't jump' and a character-level n-gram lm trained only on alphabets,
    P_lm(don't jump) = P_lm(don) * P_lm(t) * P_lm(jump).

    Args:
        beam_size (int): Max size of the beam at each timestep. (default: 50)
        max_labels_per_timestep (int): If positive, labels at each timestep are
            ranked by their scores and only the specified number of highest
            scoring labels are considered for the beam update. Otherwise, all
            output labels are considered. (default: -1)
        lm_path (str): Path to optional KenLM n-gram language model file
            in ARPA or binary format. (default: ``None``)
        lm_weight (float): Weight of the language model scores relative to the
            emission probabilities. (default: 1.2)
        insertion_bonus (float): Character insertion bonus to prevent favoring
            shorter length decodings since LM down-weighting doesn't occur
            during certain steps of the algorithm (blanks and repeats).
            Ref https://distill.pub/2017/ctc/. (default: 1.5)
        delete_key (str): Optional key for deletion/backspace in the
            character set if applicable. (default: "Key.backspace")
    """

    EOW: ClassVar[str] = "</s>"  # KenLM EOS token, used here as end-of-word
    OOV: ClassVar[str] = "<unk>"  # KenLM out-of-vocabulary (OOV) token

    beam_size: int = 50
    max_labels_per_timestep: int = -1
    lm_path: str | None = None
    lm_weight: float = 2.0
    insertion_bonus: float = 2.0
    delete_key: str | None = "Key.backspace"

    def __post_init__(self) -> None:
        # Initialize language model if provided
        self.lm: kenlm.Model | None = None
        if self.lm_path is not None:
            self.lm = kenlm.Model(self.lm_path)

            # KenLM state corresponding to beginning-of-sentence token <s>, but
            # actually meaning beginning-of-word (BOW) in our usage.
            self.lm_state_bow = kenlm.State()
            self.lm.BeginSentenceWrite(self.lm_state_bow)

            # Score for out-of-vocab (OOV) tokens to be used as a baseline.
            # This prevents within-vocab tokens from getting unfairly
            # down-weighted with each application of LM compared to OOV tokens.
            # We rely on the fact that KenLM interpolates unigrams with the
            # uniform distribution `backoff(null) / |vocab|` for OOV tokens.
            # We set bos and eos to False while computing the score below
            # so that the OOV score is equal to that of the unigram '<unk>'
            # and not the trigram '<s><unk></s>'.
            self.oov_score = self.lm.score(self.OOV, bos=False, eos=False)

        self.delete_label: int | None = None
        if self.delete_key is not None:
            self.delete_label = self._charset.key_to_label(self.delete_key)

        self.reset()

    def reset(self) -> None:
        # Reset the beam with empty sequence
        self.beam = [BeamState.init(self._charset.null_class, lm=self.lm)]

    def is_delete_label(self, label: int) -> bool:
        return self.delete_label is not None and label == self.delete_label

    def get_best_decodings(self, k: int = 5) -> list[tuple[Any, Any]]:
        # self.beam is already sorted
        return [(b.decoding, b.timestamps) for b in self.beam[:k]]

    def decode(
        self,
        emissions: np.ndarray,
        timestamps: np.ndarray,
        finish: bool = False,
    ) -> LabelData:
        assert emissions.ndim == 2  # (T, num_classes)
        assert emissions.shape[1] == self._charset.num_classes
        assert len(emissions) == len(timestamps)

        # Sort and filter label probs to only consider the highest scoring
        # `max_labels_per_timestep` labels at each timestep.
        indices = np.argsort(-emissions, axis=1)
        if self.max_labels_per_timestep > 0:
            indices = indices[:, : self.max_labels_per_timestep]

        for t in range(len(emissions)):
            # Dict to store the next set of candidate beams
            next_beam: dict[Any, BeamState] = {}

            for prev in self.beam:  # Loop over current candidates
                # CTC blank or repeat scenario
                next_ = self.next_state(
                    prev_state=prev, label=None, timestamp=None, cache=next_beam
                )

                for label in indices[t]:  # Loop over labels at time t
                    p = emissions[t, label]
                    timestamp = timestamps[t]

                    if label == self._charset.null_class:  # Blank label
                        next_.p_b = logsumexp(next_.p_b, prev.p_b + p, prev.p_nb + p)
                        continue

                    next_n = self.next_state(
                        prev_state=prev,
                        label=label,
                        timestamp=timestamp,
                        cache=next_beam,
                    )
                    p_lm = self.lm_score(prev, next_n)

                    if label == prev.label:
                        next_.p_nb = logsumexp(next_.p_nb, prev.p_nb + p)
                        next_n.p_nb = logsumexp(next_n.p_nb, prev.p_b + p + p_lm)
                    else:
                        next_n.p_nb = logsumexp(
                            next_n.p_nb, prev.p_b + p + p_lm, prev.p_nb + p + p_lm
                        )

            self.beam = sorted(
                next_beam.values(), key=lambda x: x.p_total, reverse=True
            )
            self.beam = self.beam[: self.beam_size]

        if finish:
            self.finish()

        return LabelData.from_labels(
            labels=self.beam[0].decoding,
            timestamps=self.beam[0].timestamps,
            _charset=self._charset,
        )

    def finish(self) -> LabelData:
        """To be called at the end of the sequence to finish any pending
        LM states by adding end-of-word </s> tokens."""
        if not self.lm:
            # Nothing to do, just return the best decoding
            return LabelData.from_labels(
                labels=self.beam[0].decoding,
                timestamps=self.beam[0].timestamps,
                _charset=self._charset,
            )

        for state in self.beam:
            if state.lm_state == self.lm_state_bow:
                continue

            # We have unfinished LM business (ref comments in `apply_lm()`).
            lm_state = kenlm.State()
            lm_score = self.lm.BaseScore(state.lm_state, self.EOW, lm_state)
            p_lm = self.lm_weight * lm_score
            state.p_b += p_lm
            state.p_nb += p_lm

        self.beam = sorted(self.beam, key=lambda x: x.p_total, reverse=True)
        return LabelData.from_labels(
            labels=self.beam[0].decoding,
            timestamps=self.beam[0].timestamps,
            _charset=self._charset,
        )

    def next_state(
        self,
        prev_state: BeamState,
        label: int | None = None,
        timestamp: Any | None = None,
        cache: dict[Any, BeamState] | None = None,
    ) -> BeamState:
        """Returns the next BeamState by extending `prev_state` with `label`
        and applying LM as appropriate.

        If `label` is None, we treat that as CTC blank label and the next state
        is the same as the previous state. Otherwise, we extend the decoding
        trie with the new label.

        The LM trie is extended if the new label doesn't correspond to a
        delete key. If it is a delete label, we backtrack up the LM trie by a
        node and the returned BeamState holds a reference to this LM node."""
        # Get the hash corresponding to extending prev state with `label`
        _hash = prev_state.hash(label)
        key = _hash.digest()
        if cache is not None and key in cache:
            return cache[key]

        if label is None:
            # CTC blank or repeat. No change in LM from prev state.
            label_node = prev_state.label_node
            lm_node = prev_state.lm_node
        elif self.lm is None:
            # No LM initialized. Just extend the label trie with new label.
            label_node = prev_state.label_node.child((label, timestamp))
            lm_node = None
        elif not self.is_delete_label(label):
            assert prev_state.lm_node is not None
            # New label is a non-delete key. Apply LM on the label with the
            # prev LM state as context, and extend the LM trie as well.
            label_node = prev_state.label_node.child((label, timestamp))
            lm_state, lm_score = self.apply_lm(prev_state.lm_state, label)
            lm_node = prev_state.lm_node.child((lm_state, lm_score))
        else:
            assert prev_state.lm_node is not None
            # New label is a delete key. Extend the label trie with the new
            # label as before, but safely backtrack up the LM trie by one node
            # to reach the LM state as a result of deleting the prev label.
            label_node = prev_state.label_node.child((label, timestamp))
            lm_node = (
                prev_state.lm_node
                if prev_state.lm_node.is_root
                else prev_state.lm_node.parent
            )

        next_state = BeamState(label_node, lm_node, _hash=_hash)
        if cache is not None:
            cache[key] = next_state
        return next_state

    def apply_lm(
        self,
        prev_lm_state: kenlm.State,
        label: int,
    ) -> tuple[kenlm.State, float]:
        """Takes in KenLM state and a token label, and returns a tuple of the
        next KenLM state on applying the token as well as the LM score.

        For tokens not in LM vocabulary, we return a default baseline score
        that is equal to the unigram `<unk>` score of the KenLM model
        corresponding to OOV tokens."""
        assert self.lm is not None
        assert not self.is_delete_label(label)

        key = self._charset.label_to_key(label)
        if key in self.lm:
            # Token is in LM vocab
            lm_state = kenlm.State()
            lm_score = self.lm.BaseScore(prev_lm_state, key, lm_state)
        elif prev_lm_state != self.lm_state_bow:
            # LM states corresponding to tokens not in LM vocab are set to
            # `self.lm_state_bow` corresponding to <s>. Therefore, if the prev
            # LM state isn't <s> and we have an OOV token now, we end the word
            # and overwrite the LM state to `self.lm_state_bow` to begin
            # the next word.
            lm_state = kenlm.State()
            lm_score = self.lm.BaseScore(prev_lm_state, self.EOW, lm_state)
            lm_state = self.lm_state_bow
        else:
            # Prev LM state corresponds to <s>, but we have an OOV token.
            # Set this LM state to <s> as well to begin the next word.
            lm_score = self.oov_score
            lm_state = self.lm_state_bow

        return lm_state, lm_score

    def lm_score(self, prev_state: BeamState, next_state: BeamState) -> float:
        """Helper to safely compute the weighted LM score to be added.

        Handles deletion scenario by undoing the LM score corresponding to the
        previous label. The invariant we want is that the total LM score of a
        sequence of labels should be the same whether or not the sequence was
        mistyped and corrected via deletes."""
        if self.lm is None:  # No LM initialized
            return 0.0

        assert prev_state.lm_node is not None

        if not self.is_delete_label(next_state.label):
            lm_score = next_state.lm_score
            p_lm = self.lm_weight * lm_score + self.insertion_bonus
        elif not prev_state.lm_node.is_root:
            # Deleting the last label, we need to undo prev LM score
            lm_score = prev_state.lm_score
            p_lm = -self.lm_weight * lm_score - self.insertion_bonus
        else:
            # Deletion case but we are the root of LM trie, meaning we are
            # attempting to delete an empty sequence. No prev LM score to undo.
            assert prev_state.lm_score == 0.0
            p_lm = 0.0

        return p_lm
