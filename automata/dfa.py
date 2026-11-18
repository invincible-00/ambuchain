"""
automata/dfa.py
===============
Custom Deterministic Finite Automaton (DFA) engine and the AmbuChain
ambulance-lifecycle DFA.  No third-party automata libraries are used.

Mathematical definition
-----------------------
A DFA is a 5-tuple  M = (Q, Σ, δ, q0, F)  where

    Q   finite, non-empty set of states
    Σ   finite input alphabet
    δ   : Q × Σ → Q   transition function
    q0  ∈ Q            start state
    F   ⊆ Q            set of accepting states

The AmbuChain ambulance DFA is

    Q  = { IDLE, CALL_LOGGED, UNIT_ASSIGNED, UNIT_MOVING, ON_SCENE,
           PATIENT_ABOARD, TRANSPORTING, AT_HOSPITAL, CLOSED }

    Σ  = { CALL_RECEIVED, AMBULANCE_DISPATCHED, EN_ROUTE, ARRIVED_SCENE,
           PATIENT_LOADED, DEPARTED_SCENE, ARRIVED_HOSPITAL,
           HANDOVER_COMPLETE }

    q0 = IDLE
    F  = { CLOSED }

    δ(q_i, e_i) = q_{i+1}   for i = 0..7   (q_i = i-th state of Q above,
                                            e_i = i-th symbol of Σ above)

Every (state, symbol) pair NOT listed in δ is an *invalid transition*.
Formally these go to an implicit dead (trap) state; call `DFA.complete()`
to materialise it and obtain a total transition function.

A sequence is
    valid     – no invalid transition occurred,
    complete  – it ended in an accepting state (F),
    accepted  – valid AND complete  (i.e. w ∈ L(M)).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

UNDEFINED_TRANSITION = "UNDEFINED_TRANSITION"
UNKNOWN_SYMBOL = "UNKNOWN_SYMBOL"


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TransitionRecord:
    """One step of a DFA run."""

    index: int                    # position of the symbol in the input
    from_state: str
    symbol: str
    to_state: Optional[str]       # None when the transition was invalid
    valid: bool
    reason: Optional[str] = None  # UNDEFINED_TRANSITION | UNKNOWN_SYMBOL

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationResult:
    """Outcome of running a symbol sequence through a DFA."""

    start_state: str
    final_state: str              # last state actually reached
    is_complete: bool             # final_state ∈ F
    halted: bool                  # run stopped early at an invalid transition
    trace: List[TransitionRecord] = field(default_factory=list)
    invalid_transitions: List[TransitionRecord] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.invalid_transitions

    @property
    def accepted(self) -> bool:
        return self.is_valid and self.is_complete

    @property
    def first_invalid(self) -> Optional[TransitionRecord]:
        return self.invalid_transitions[0] if self.invalid_transitions else None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly representation (handy for the frontend)."""
        return {
            "start_state": self.start_state,
            "final_state": self.final_state,
            "is_valid": self.is_valid,
            "is_complete": self.is_complete,
            "accepted": self.accepted,
            "halted": self.halted,
            "trace": [t.to_dict() for t in self.trace],
            "invalid_transitions": [t.to_dict() for t in self.invalid_transitions],
        }


# --------------------------------------------------------------------------- #
# Generic DFA
# --------------------------------------------------------------------------- #
class DFA:
    """A generic DFA  M = (Q, Σ, δ, q0, F).

    `transitions` maps (state, symbol) -> state and may be partial; any
    missing pair is treated as an invalid transition.
    """

    def __init__(
        self,
        states: Iterable[str],
        alphabet: Iterable[str],
        transitions: Dict[Tuple[str, str], str],
        start_state: str,
        accepting_states: Iterable[str],
    ) -> None:
        self.states = frozenset(states)
        self.alphabet = frozenset(alphabet)
        self.transitions = dict(transitions)
        self.start_state = start_state
        self.accepting_states = frozenset(accepting_states)
        self._check_definition()

    # -- definition ---------------------------------------------------------
    def _check_definition(self) -> None:
        if not self.states:
            raise ValueError("Q (states) must be non-empty.")
        if self.start_state not in self.states:
            raise ValueError(f"Start state {self.start_state!r} is not in Q.")
        if not self.accepting_states <= self.states:
            raise ValueError("F (accepting states) must be a subset of Q.")
        for (state, symbol), target in self.transitions.items():
            if state not in self.states or target not in self.states:
                raise ValueError(f"Transition ({state!r}, {symbol!r}) -> {target!r} uses a state outside Q.")
            if symbol not in self.alphabet:
                raise ValueError(f"Transition symbol {symbol!r} is not in Σ.")

    # -- primitives ---------------------------------------------------------
    def next_state(self, state: str, symbol: str) -> Optional[str]:
        """δ(state, symbol), or None if the transition is undefined."""
        return self.transitions.get((state, symbol))

    def is_valid_transition(self, state: str, symbol: str) -> bool:
        return (state, symbol) in self.transitions

    def valid_symbols_from(self, state: str) -> List[str]:
        """Symbols that have a defined transition out of `state`."""
        return [s for (q, s) in self.transitions if q == state]

    # -- execution ----------------------------------------------------------
    def run(self, symbols: Sequence[str], stop_on_error: bool = True) -> ValidationResult:
        """Run `symbols` through the DFA.

        stop_on_error=True  : classic DFA behaviour - halt at the first invalid
                              transition (equivalent to entering the trap state).
        stop_on_error=False : recovery mode - record the invalid transition,
                              skip that symbol (state unchanged) and keep going,
                              so ALL invalid transitions can be listed.
                              Note: one skipped event can cascade into more.
        """
        state = self.start_state
        trace: List[TransitionRecord] = []
        invalid: List[TransitionRecord] = []
        halted = False

        for i, symbol in enumerate(symbols):
            if symbol not in self.alphabet:
                rec = TransitionRecord(i, state, symbol, None, False, UNKNOWN_SYMBOL)
            else:
                target = self.next_state(state, symbol)
                if target is None:
                    rec = TransitionRecord(i, state, symbol, None, False, UNDEFINED_TRANSITION)
                else:
                    rec = TransitionRecord(i, state, symbol, target, True)
                    state = target
            trace.append(rec)
            if not rec.valid:
                invalid.append(rec)
                if stop_on_error:
                    halted = True
                    break

        return ValidationResult(
            start_state=self.start_state,
            final_state=state,
            is_complete=state in self.accepting_states,
            halted=halted,
            trace=trace,
            invalid_transitions=invalid,
        )

    def accepts(self, symbols: Sequence[str]) -> bool:
        """True iff the sequence is in L(M): valid and ends in F."""
        return self.run(symbols).accepted

    # -- utilities ----------------------------------------------------------
    def complete(self, trap_state: str = "TRAP") -> "DFA":
        """Return an equivalent DFA with a total δ (explicit dead state)."""
        if trap_state in self.states:
            raise ValueError(f"{trap_state!r} already exists in Q.")
        full = dict(self.transitions)
        all_states = set(self.states) | {trap_state}
        for q in all_states:
            for s in self.alphabet:
                full.setdefault((q, s), trap_state)
        return DFA(all_states, self.alphabet, full, self.start_state, self.accepting_states)

    def transition_table(self) -> List[Tuple[str, str, str]]:
        """Defined transitions as (from_state, symbol, to_state) rows."""
        return [(q, s, t) for (q, s), t in self.transitions.items()]


# --------------------------------------------------------------------------- #
# AmbuChain ambulance DFA
# --------------------------------------------------------------------------- #
STATES: Tuple[str, ...] = (
    "IDLE", "CALL_LOGGED", "UNIT_ASSIGNED", "UNIT_MOVING", "ON_SCENE",
    "PATIENT_ABOARD", "TRANSPORTING", "AT_HOSPITAL", "CLOSED",
)
EVENTS: Tuple[str, ...] = (
    "CALL_RECEIVED", "AMBULANCE_DISPATCHED", "EN_ROUTE", "ARRIVED_SCENE",
    "PATIENT_LOADED", "DEPARTED_SCENE", "ARRIVED_HOSPITAL", "HANDOVER_COMPLETE",
)
START_STATE = STATES[0]
ACCEPTING_STATES = (STATES[-1],)

STATE_INDEX: Dict[str, int] = {s: i for i, s in enumerate(STATES)}
EVENT_INDEX: Dict[str, int] = {e: i for i, e in enumerate(EVENTS)}


def build_ambulance_dfa() -> DFA:
    """Build the ambulance DFA:  δ(STATES[i], EVENTS[i]) = STATES[i+1]."""
    transitions = {(STATES[i], EVENTS[i]): STATES[i + 1] for i in range(len(EVENTS))}
    return DFA(STATES, EVENTS, transitions, START_STATE, ACCEPTING_STATES)


AMBULANCE_DFA: DFA = build_ambulance_dfa()


def normalize_event(event: Any) -> str:
    """'  en route ' -> 'EN_ROUTE'.  Keeps matching tolerant of messy input."""
    return str(event).strip().upper().replace(" ", "_")


def validate_event_sequence(events: Sequence[Any], stop_on_error: bool = True) -> ValidationResult:
    """Validate one incident's ordered event names against the ambulance DFA."""
    return AMBULANCE_DFA.run([normalize_event(e) for e in events], stop_on_error=stop_on_error)


def is_valid_sequence(events: Sequence[Any]) -> bool:
    """True iff the events form a complete, correctly ordered lifecycle."""
    return validate_event_sequence(events).accepted


def find_invalid_transitions(events: Sequence[Any]) -> List[Dict[str, Any]]:
    """All invalid transitions (recovery mode), as plain dicts."""
    result = validate_event_sequence(events, stop_on_error=False)
    return [t.to_dict() for t in result.invalid_transitions]


def get_expected_event(state: str) -> Optional[str]:
    """The only event the DFA accepts from `state` (None in the final state)."""
    pos = STATE_INDEX[state]
    return EVENTS[pos] if pos < len(EVENTS) else None


def get_dfa_definition() -> Dict[str, Any]:
    """The formal 5-tuple as plain data, for display in the frontend."""
    return {
        "Q": list(STATES),
        "Sigma": list(EVENTS),
        "q0": START_STATE,
        "F": list(ACCEPTING_STATES),
        "delta": [
            {"from": q, "event": s, "to": t} for q, s, t in AMBULANCE_DFA.transition_table()
        ],
        "note": "Any (state, event) pair not in delta is an invalid transition (implicit trap state).",
    }
