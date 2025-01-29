# I just tried to implement L* using my own understanding.

# First assume there is a DFA we want to learn. It should stay "secret"
# or in other words, it is a black box

import re
# A dummy example
# Any binary string contain the substring 01
DUMMY_DFA = re.compile(r'^(0|1)*01(0|1)*$')


# DUMMY_DFA = re.compile(r'^(11|00|(10|01)(11|00)(10|01))$')

def L_star(alphabet:list[str]):
    Q = {""}
    T = {""}
    M = {}
    while True:
        make_consistent(Q, T, M, alphabet,membership_oracle)
        make_closed(Q,T ,M,alphabet,membership_oracle)
        dfa = build_hypothesis_dfa(Q, T, M, alphabet)
        print("current hypothesis dfa", dfa)
        counterexample = find_counterexample(dfa, membership_oracle, alphabet, max_length=8)
        print("counterexample", counterexample)
        if counterexample is None:
            return dfa
        else:
            for i in range(1, len(counterexample) + 1):
                pref = counterexample[:i]
                if pref not in Q:
                    Q.add(pref)


def make_consistent(Q, T, M, alphabet,membership_oracle) -> None:
    while True:
        consistent = True
        q_list = list(Q)
        for i in range(len(q_list)):
            for j in range(i + 1, len(q_list)):
                q1 = q_list[i]
                q2 = q_list[j]
                # If row(q1) == row(q2)
                if is_T_equivalent(q1, q2, M, T,membership_oracle):
                    for a in alphabet:
                        e = find_the_not_equivalent_test_word(q1 + a, q2 + a, M, T,membership_oracle)
                        if e is not None:
                            T.add(a+e)
                            for q in Q:
                                get_membership(q, a+e, M, membership_oracle)
                            consistent = False
                            break
                if not consistent:
                    break
            if not consistent:
                break
        if consistent:
            break

def make_closed(Q, T, M,alphabet:list[str],membership_oracle) -> None:
    while True:
        closed = True
        # Using the definition here
        # for every q in Q and a in the alphabet,
        # there is some q' in Q such that qa =QM q'
        for q in list(Q):  # We convert to list to avoid mutating during iteration
            for a in alphabet:
                qa = q + a
                if not any(is_T_equivalent(qa, q_prime, M, T,membership_oracle) for q_prime in Q):
                    Q.add(qa)
                    for u in T:
                        get_membership(qa, u, M, membership_oracle)
                    closed = False
                    break
            if not closed:
                # If we added, re-check from the start
                break
        if closed:
            break


def find_the_not_equivalent_test_word(v,w, M, T, membership_oracle):
    # v == w iff vu and wu are in the language for u in T
    for u in list(T):
        vu = get_membership(v,u,M,membership_oracle)
        wu = get_membership(w,u,M,membership_oracle)
        if vu != wu:
            return u
    return None

def is_T_equivalent(v,w, M, T, membership_oracle):
    if find_the_not_equivalent_test_word(v,w, M, T, membership_oracle) is not None:
        return False
    return True

def get_membership(q, t, M, membership_oracle):
    if q not in M:
        M[q] = {}
    if t not in M[q]:
        M[q][t] = membership_oracle(q + t)
    return M[q][t]

def membership_oracle(word):
    # If the word is in the language return True
    return bool(DUMMY_DFA.match(word))

def row_signature(q, T, M):
    # Sort T just to have a consistent ordering.
    sorted_T = sorted(T)
    return tuple(M[q][u] for u in sorted_T)

def build_hypothesis_dfa(Q, T, M, alphabet):
    signature_map = {}
    for q in Q:
        signature = row_signature(q, T, M)
        signature_map[q] = signature

    sig_to_rep = {}
    rep_index = {}
    states = []

    for q in Q:
        sig = signature_map[q]
        if sig not in sig_to_rep:
            sig_to_rep[sig] = q

    state_id = 0
    for sig, rep in sig_to_rep.items():
        rep_index[rep] = state_id
        states.append(rep)
        state_id += 1

    start_state = rep_index[""]

    accepting_states = set()
    for rep in rep_index:
        if M[rep][""]:
            accepting_states.add(rep_index[rep])

    transitions = {}
    for rep in rep_index:
        s_id = rep_index[rep]
        transitions[s_id] = {}
        for a in alphabet:
            next_prefix = rep + a
            if next_prefix not in Q:
                matched_q = None
                for q_prime in Q:
                    if is_T_equivalent(next_prefix, q_prime, M, T, membership_oracle):
                        matched_q = q_prime
                        break
                if matched_q is None:
                    Q.add(next_prefix)
                    matched_q = next_prefix
                    for u in T:
                        get_membership(M, next_prefix, u, membership_oracle)
                next_prefix = matched_q
            next_sig = signature_map[next_prefix]
            next_rep = sig_to_rep[next_sig]
            next_state = rep_index[next_rep]
            transitions[s_id][a] = next_state

    return {
        "num_states": len(states),
        "alphabet": alphabet,
        "states": states,  # each index i is the representative prefix for state i
        "start_state": start_state,
        "accepting_states": accepting_states,
        "transitions": transitions
    }

def find_counterexample(dfa, membership_oracle, alphabet, max_length=8):
    from collections import deque

    # BFS
    queue = deque([""])
    while queue:
        s = queue.popleft()
        dfa_accepts = dfa_accept(dfa, s)
        oracle_accepts = membership_oracle(s)

        if dfa_accepts != oracle_accepts:
            return s

        if len(s) < max_length:
            for a in alphabet:
                queue.append(s + a)
    return None

def dfa_accept(dfa, s):
    current_state = dfa["start_state"]
    for ch in s:
        current_state = dfa["transitions"][current_state][ch]
    return current_state in dfa["accepting_states"]

if __name__ == "__main__":
    alphabet = ["0", "1"]
    learned_dfa = L_star(alphabet)
    print("Learned DFA:", learned_dfa)