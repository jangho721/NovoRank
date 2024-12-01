import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


amino_acid = {'G': 57.02146372376, 'A': 71.03711378804, 'S': 87.03202841014, 'P': 97.05276385232, 'V': 99.0684139166,
              'T': 101.04767847442, 'C': 160.03064447804, 'L': 113.08406398088, 'I': 113.08406398088, 'N': 114.04292744752,
              'D': 115.02694303224, 'Q': 128.0585775118, 'K': 128.09496301826, 'E': 129.04259309652, 'M': 131.0404846066,
              'H': 137.0589118628, 'F': 147.0684139166, 'R': 156.10111102874, 'Y': 163.0633285387, 'W': 186.07931295398,
              'm': 147.0353846066}

h20 = 18.01528
nh3 = 17.02655
proton = 1.00727647
neutron = 1.0086710869

fragment_tolerance = 0.025


class FeatureTransformer:
    def __init__(self, dataset):
        self.dataset = dataset

    def apply_log_scaling(self):

        """
        Apply logarithmic scaling to the 'Score' and 'New Count' columns to normalize the values.
        The transformed 'Score' is then divided by the transformed 'New Count' for scaling.
        """

        logging.info('Starting log scaling process.')

        # Apply log transformation to 'Score' and 'New Count'
        self.dataset['Score'] = np.log(self.dataset['Score'] + 1)
        self.dataset['New Count'] = np.log(self.dataset['New Count'] + 1)

        # Normalize 'Score' by dividing it by 'New Count'
        self.dataset['Score'] = self.dataset['Score'] / self.dataset['New Count']

        logging.info('Log scaling process completed successfully.')

    def calculate_delta_scores(self):

        """
        Calculate the delta scores for each row.

        Returns:
            list: A list of delta scores corresponding to each row in the dataset.
        """

        logging.info('Starting delta score calculation process.')

        delta_scores = []  # List to store the calculated delta scores
        previous_key = None  # To store the previous key
        highest_score = 0  # To track the highest score

        for source_file, scan_number, score in self.dataset[['Source File', 'Scan number', 'Score']].values:
            current_key = (source_file, scan_number)

            if current_key == previous_key:
                # If still in the same group, calculate the delta score
                delta_scores.append(highest_score - score)
            else:
                # If a new group starts, initialize delta score to 0
                delta_scores.append(0)
                highest_score = score  # Reset highest score for the new group

            previous_key = current_key  # Update the current key
            highest_score = max(highest_score, score)  # Update the highest score

        logging.info('Delta score calculation process completed successfully.')

        return delta_scores

    def generate_features(self):

        """
        Generate new features for the dataset by applying log scaling and calculating delta scores.
        The results are added as new columns to the dataset.
        """

        # Apply log scaling to the dataset
        self.apply_log_scaling()

        # Calculate delta scores and add them as a new column 'Delta Score'
        self.dataset['Delta Score'] = self.calculate_delta_scores()

        # Reset the index
        self.dataset.reset_index(drop=True)
