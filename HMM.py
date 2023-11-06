import random
import argparse
import codecs
import os
import numpy as np

# observations
class Observation:
    def __init__(self, stateseq, outputseq):
        self.stateseq  = stateseq   # sequence of states
        self.outputseq = outputseq  # sequence of outputs
    def __str__(self):
        return ' '.join(self.stateseq)+'\n'+' '.join(self.outputseq)+'\n'
    def __repr__(self):
        return self.__str__()
    def __len__(self):
        return len(self.outputseq)

# hmm model
class HMM:
    def __init__(self, transitions={}, emissions={}):
        """creates a model from transition and emission probabilities"""
        ## Both of these are dictionaries of dictionaries. e.g. :
        # {'#': {'C': 0.814506898514, 'V': 0.185493101486},
        #  'C': {'C': 0.625840873591, 'V': 0.374159126409},
        #  'V': {'C': 0.603126993184, 'V': 0.396873006816}}

        self.transitions = transitions
        self.emissions = emissions

    ## part 1 - you do this.
    def load(self, basename):
        """reads HMM structure from transition (basename.trans),
        and emission (basename.emit) files,
        as well as the probabilities."""
        
        with open(f'{basename}.trans', 'r') as trans_file:
            lines = trans_file.read().splitlines()
            for line in lines:
                parts = line.split()
                if len(parts) == 3:
                    from_state, to_state, probability = parts
                    if from_state not in self.transitions:
                        self.transitions[from_state] = {}
                    self.transitions[from_state][to_state] = float(probability)

        with open(f'{basename}.emit', 'r') as emit_file:
            lines = emit_file.read().splitlines()
            for line in lines:
                parts = line.split()
                if len(parts) == 3:
                    state, output, probability = parts
                    if state not in self.emissions:
                        self.emissions[state] = {}
                    self.emissions[state][output] = float(probability)



   ## you do this.
    def generate(self, n):
        """return an n-length observation by randomly sampling from this HMM."""
        
        observation = Observation([], [])

        current_state = '#'

        for _ in range(n):
            next_state = random.choices(
                list(self.transitions[current_state].keys()),
                weights=list(self.transitions[current_state].values())
            )[0]

            emission = random.choices(
                list(self.emissions[next_state].keys()),
                weights=list(self.emissions[next_state].values())
            )[0]

            observation.stateseq.append(next_state)
            observation.outputseq.append(emission)

            current_state = next_state

        return observation

    def forward(self, observation):
        # Initialize the forward probabilities
        forward_probabilities = {state: 0.0 for state in self.transitions}

        # Initialize the forward probabilities at the starting state
        forward_probabilities['#'] = 1.0

        # Iterate through the observation sequence
        for output in observation.outputseq:
            new_forward_probabilities = {state: 0.0 for state in self.transitions}
            for to_state in self.transitions:
                for from_state in self.transitions:
                    transition_probability = self.transitions[from_state][to_state]
                    emission_probability = self.emissions[to_state].get(output, 0.0)
                    new_forward_probabilities[to_state] += forward_probabilities[from_state] * transition_probability * emission_probability

            forward_probabilities = new_forward_probabilities

        # Find the most likely final state
        final_state = max(forward_probabilities, key=forward_probabilities.get)

        return final_state


    ## you do this: Implement the Viterbi alborithm. Given an Observation (a list of outputs or emissions)
    ## determine the most likely sequence of states.

    def viterbi(model, observation):
        n = len(observation.outputseq)
        num_states = len(model.transitions)

        if n == 0:
            return []  # Empty observation sequence, no states to predict

        viterbi_matrix = np.zeros((num_states, n), dtype=float)
        backpointers = np.zeros((num_states, n), dtype=int)

        # Initialize the Viterbi matrix at time t=0
        for i, state in enumerate(model.transitions):
            viterbi_matrix[i, 0] = (
                model.transitions['#'].get(state, 0.0) *
                model.emissions[state].get(observation.outputseq[0], 0.0)
            )

        # Fill in the Viterbi matrix using dynamic programming
        for t in range(1, n):
            for i, to_state in enumerate(model.transitions):
                max_probability, best_previous_state = max(
                    (viterbi_matrix[j, t - 1] *
                    model.transitions[from_state].get(to_state, 0.0) *
                    model.emissions[to_state].get(observation.outputseq[t], 0.0),
                    j) for j, from_state in enumerate(model.transitions))
                viterbi_matrix[i, t] = max_probability
                backpointers[i, t] = best_previous_state

        # Backtrack to find the most likely sequence of states
        final_state_index = np.argmax(viterbi_matrix[:, n - 1])
        state_sequence = [list(model.transitions.keys())[final_state_index]]
        for t in range(n - 1, 0, -1):
            final_state_index = backpointers[final_state_index, t]
            state_sequence.insert(0, list(model.transitions.keys())[final_state_index])

        return state_sequence





