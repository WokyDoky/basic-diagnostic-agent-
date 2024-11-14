import numpy as np
import pandas as pd
from typing import List


class DiagnosticAgent:
    def __init__(self):
        # Initialize 5D array for joint probability distribution
        # Dimensions: [Symptom1, Symptom2, Symptom3, Symptom4, Condition]
        # Symptoms: 2 possible values (0,1)
        # Condition: 4 possible values (0,1,2,3)
        self.joint_dist = np.zeros((2, 2, 2, 2, 4))
        self.total_samples = 0

    def learn_distribution(self, data: pd.DataFrame) -> None:
        """
        Learn the joint probability distribution from data by counting and normalizing.

        Args:
            data: DataFrame with columns [Symptom1, Symptom2, Symptom3, Symptom4, Condition]
        """
        self.total_samples = len(data)

        # Count occurrences for each combination
        for _, row in data.iterrows():
            s1, s2, s3, s4, c = row.values
            self.joint_dist[int(s1), int(s2), int(s3), int(s4), int(c)] += 1

        # Normalize to get probabilities
        self.joint_dist /= self.total_samples

    def inference_by_enumeration(self, evidence: List[int]) -> np.ndarray:
        """
        Perform inference by enumeration.

        Args:
            evidence: List of 5 values where:
                     -2 = query variable
                     -1 = hidden variable
                     0,1 = evidence for symptoms
                     0,1,2,3 = evidence for condition

        Returns:
            Probability distribution over the query variable
        """
        # Find query variable index
        query_idx = evidence.index(-2)

        # Initialize result array
        if query_idx == 4:  # If condition is query
            result = np.zeros(4)
        else:  # If symptom is queried
            result = np.zeros(2)

        # Get dimensions for iteration
        dims = [2, 2, 2, 2, 4]

        # Create an array of indices to iterate over
        index_arrays = []
        for i, (e, dim) in enumerate(zip(evidence, dims)):
            if e == -2 or e == -1:  # Query or hidden
                index_arrays.append(range(dim))
            else:  # Evidence
                index_arrays.append([e])

        # Enumerate all possible combinations
        for indices in np.ndindex(*[len(arr) for arr in index_arrays]):
            # Map to actual values
            actual_indices = tuple(arr[idx] for arr, idx in zip(index_arrays, indices))

            # Add probability to result
            prob = self.joint_dist[actual_indices]
            if prob > 0:  # Only consider non-zero probabilities
                result[actual_indices[query_idx]] += prob

        # Normalize
        if np.sum(result) > 0:
            result = result / np.sum(result)

        return result

    def print_probabilities(self, evidence: List[int]) -> None:
        """
        Print the probability distribution for the query variable given the evidence.

        Args:
            evidence: List of five values representing the evidence array.
        """
        prob_dist = self.inference_by_enumeration(evidence)

        for i, prob in enumerate(prob_dist):
            print(f"Condition {i}: {prob:.3f}")

    def calculate_marginals(self) -> None:
        """
        Calculate and print the marginal probabilities for each symptom and the condition.
        """

        # Loop through each variable to calculate its marginal probability
        for i in range(5):
            # Create an evidence array with all -1 (hidden), except -2 for the query
            evidence = [-1] * 5
            evidence[i] = -2  # Set the current variable as the query

            # Calculate marginal probability for the current variable
            marginal_prob = self.inference_by_enumeration(evidence)

            # Print results
            if i < 4:  # For symptoms (0 to 3)
                print(f"Marginal probability for Symptom{i + 1}:")
                for j, prob in enumerate(marginal_prob):
                    print(f"  P(Symptom{i + 1}={j}): {prob:.3f}")
            else:  # For condition (4th index)
                print("Marginal probability for Condition:")
                for j, prob in enumerate(marginal_prob):
                    print(f"  P(Condition={j}): {prob:.3f}")
            print()  # Blank line for readability



def main():
    # Read data
    agent = DiagnosticAgent()
    try:
        data = pd.read_csv('Health_Data_Set.csv')
        agent.learn_distribution(data)
    except FileNotFoundError:
        print("File 'Health_Data_Set.csv' Not Found.")
        return

    # Testing queries:

    #=======================================================

    """
        # 1. P(Condition | Symptom1=1, Symptom2=1)
    evidence = [1, 1, -1, -1, -2]  # -2 for a query, -1 for hidden, actual values for evidence
    prob_dist = agent.inference_by_enumeration(evidence)
    print("\nP(Condition | Symptom1=1, Symptom2=1):")
    for i, prob in enumerate(prob_dist):
        print(f"Condition {i}: {prob:.3f}")

    # 2. P(Symptom3 | Condition=1, Symptom1=1)
    evidence = [1, -1, -2, -1, 1]
    prob_dist = agent.inference_by_enumeration(evidence)
    print("\nP(Symptom3 | Condition=1, Symptom1=1):")
    for i, prob in enumerate(prob_dist):
        print(f"Symptom3={i}: {prob:.3f}")
    """

    #--------------------------------------------------------

    profiles = [
        [1, 0, 1, 1, -2],  # Profile a
        [1, -1, 1, 0, -2],  # Profile b
        [1, -1, 0, 1, -2],  # Profile c
        [-1, 1, -1, -1, -2],  # Profile d
        [-1, 1, -1, 0, -2],  # Profile e
        [1, 1, -1, -1, -2]  # Profile f
    ]

    # Print probabilities for each profile
    for i, evidence in enumerate(profiles, start=1):
        print(f"\nProfile {chr(96 + i)}:")
        agent.print_probabilities(evidence)

    print()
    agent.calculate_marginals()
if __name__ == "__main__":
    main()