import os
import logging
import numpy as np

from numba import jit
from tqdm import tqdm
from itertools import accumulate

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
        return self.dataset.reset_index(drop=True)


class featureExtractor:
    def __init__(self, config):

        """
        Initialize the feature extractor with configuration settings
        """

        self.denoised_mgf_path = config['path']['denoised_mgf_path']

    def internal_fragment_ion_features (self, dataset):

        """
        Process the dataset to calculate internal fragment ion features
        """

        # Separate data into those with missing sequences and those with sequences
        dataset_missing_sequence = dataset[dataset['Sequence'].isnull()]
        dataset_with_sequence = dataset[dataset['Sequence'].notnull()]

        # Calculate internal fragment ions and sequence lengths
        dataset_with_sequence = self.calculate_internal_fragment_ion(dataset_with_sequence)
        dataset_with_sequence = self.calculate_sequence_length(dataset_with_sequence)

        # Normalize internal fragment ion counts by sequence length
        dataset_with_sequence['Normalized Internal Fragment Ions'] = dataset_with_sequence['Internal Fragment Ions Count'] / dataset_with_sequence['Sequence Length']
        dataset_with_sequence['Normalized Internal Fragment Ions'] = np.log(dataset_with_sequence['Normalized Internal Fragment Ions'] + 1)

        # Handle missing sequence data by setting related columns to NaN
        dataset_missing_sequence[['Internal Fragment Ions Count', 'Sequence Length', 'Normalized Internal Fragment Ions']] = np.nan

        # Merge the datasets and sort by specified columns
        dataset_combined = pd.concat([dataset_with_sequence, dataset_missing_sequence])
        dataset_combined = dataset_combined.sort_values(by=['Source File', 'Scan number', 'Rank']).reset_index(drop=True)

        return dataset_combined

    def calculate_internal_fragment_ion(self, dataset):

        """
       Calculate the count of internal fragment ions for each entry in the dataset
       using the provided denoised .mgf files.
       """

        # Sort the list of .mgf files in the denoised path
        file_list = sorted(os.listdir(self.denoised_mgf_path))

        # Initialize variables
        current_file_index, line_index = 0, 0
        current_file_data, current_spec_key = None, None
        experimental_peak_list = []
        internal_fragment_ion = []

        process_ion_data = self.process_ion_data
        denoised_mgf_path = self.denoised_mgf_path

        # Iterate over each row of the dataset
        for source_file, scan_number, peptide, charge, mz_value in tqdm(
                dataset[['Source File', 'Scan number', 'Peptide', 'Charge', 'Mass/Charge (m/z)']].values):

            # Extract base name and scan number for spectrum identification
            file_base_name = source_file.split('.')[0]
            scan_number = int(scan_number)
            spec_key = (file_base_name, scan_number)

            # Load data if not already loaded, then initialize variables for the current file
            if current_file_data is None:
                current_file_name = file_list[current_file_index]
                current_file_data = open(f"{denoised_mgf_path}/{current_file_name}").readlines()
                line_index = 0

            # If processing the same spectrum, calculate and append the ion count
            if spec_key == current_spec_key:
                ion_count = process_ion_data(experimental_peak_list, charge, mz_value, peptide)
                internal_fragment_ion.append(ion_count)
                continue

            # Loop through the lines of the current .mgf file
            while True:
                # If the end of the file is reached, move to the next file
                if line_index >= len(current_file_data):
                    current_file_index += 1
                    if current_file_index >= len(file_list):  # No more files to process
                        logging.info("Reached the end of the files. No more data.")
                        break
                    current_file_name = file_list[current_file_index]
                    current_file_data = open(f"{denoised_mgf_path}/{current_file_name}").readlines()
                    line_index = 0

                # Process the current line in the .mgf file
                current_line = current_file_data[line_index].strip()  # Remove whitespace from both ends of the string

                if current_line.startswith('TITLE='):  # Identify spectrum titles

                    # Extract scan number and file name from the title
                    current_scan_number = int(current_line.split('.')[1])
                    current_file_name = current_line.split('.')[0].split('=')[1]
                    current_spec_key = (current_file_name, current_scan_number)

                    # Skip to the next file if current file is alphabetically less than the base name
                    if file_base_name > current_file_name:
                        current_file_index += 1
                        if current_file_index >= len(file_list):
                            logging.info("Reached the end of the files. No more data.")
                            break
                        current_file_name = file_list[current_file_index]
                        current_file_data = open(f"{denoised_mgf_path}/{current_file_name}").readlines()
                        line_index = 0
                        continue

                    # Initialize the experimental_peak_list
                    if spec_key == current_spec_key:
                        experimental_peak_list = []

                if spec_key == current_spec_key:
                    if current_line.startswith('END IONS'):  # Identify the end of spectrum data
                        # calculate and append the ion count
                        ion_count = process_ion_data(experimental_peak_list, charge, mz_value, peptide)
                        internal_fragment_ion.append(ion_count)
                        line_index += 1
                        break

                    # Collect peak data (mass/charge ratios)
                    if current_line.strip() and current_line.split()[0].replace('.', '', 1).isdigit():
                        mz = float(current_line.split()[0])
                        experimental_peak_list.append(mz)

                # Increment the line index
                line_index += 1

        # Add the calculated internal fragment ion counts to the dataset
        dataset['Internal Fragment Ions Count'] = internal_fragment_ion

        return dataset

    def process_ion_data(self, experimental_peaks, charge, precursor_mass, peptide):

        """
        Processes internal fragment ion data for a given peptide.

        Parameters:
            experimental_peaks (list): List of m/z values from the spectrum.
            charge (int): Charge state.
            precursor_mass (float): Precursor mass-to-charge ratio (m/z).
            peptide (str): Amino acid sequence of the peptide.

        Returns:
            int: Count of internal fragment ions.
        """

        # Calculate the b and y ion series masses for the given peptide sequence
        b_ion_list, y_ion_list = self.calculate_ion_series_masses(peptide)

        # Calculate the all possible theoretical ions
        theoretical_ion_list = self.calculate_residue_ions(b_ion_list, y_ion_list, int(charge), float(precursor_mass))

        # Remove peaks from the experimental data that match the calculated theoretical ions
        filtered_experimental_peaks = self.remove_matching_peaks(experimental_peaks, theoretical_ion_list)

        # Calculate the masses of all possible internal fragments of the peptide
        fragment_ion_mass_list = self.calculate_subpeptide_masses(peptide)

        # Consider charge effects on internal fragment ions
        internal_fragment_candidates = self.consider_charge(fragment_ion_mass_list, int(charge))

        # Count the final internal fragment ions present in the filtered experimental peaks
        ion_count = self.count_final_internal_fragments(filtered_experimental_peaks, internal_fragment_candidates)

        return ion_count

    @staticmethod
    def calculate_sequence_length(dataset):

        """
        Calculate the length of the peptide sequence for each entry in the dataset.
        """

        # Compute the length of the peptide sequence for non-null entries
        dataset['Sequence Length'] = dataset['Sequence'].apply(lambda x: len(x))
        # apply(lambda x: len(x) if pd.notnull(x) else np.nan) -> Sequence: Nan, None 처리)

        return dataset

    @staticmethod
    @jit(nopython=False, cache=True)
    def calculate_isotopes(masses):

        """
        Calculate isotopic shifts for a list of masses.

        Parameters:
            masses (np.array): Array of base masses.

        Returns:
            np.array: Array containing base masses along with +1 and +2 neutron shifts.
        """

        n = len(masses)
        isotopes = np.empty(n * 3)
        for i in range(n):
            base_mass = masses[i]

            # Include the base mass and isotopic shifts (+1 neutron, +2 neutrons)
            isotopes[i * 3] = base_mass
            isotopes[i * 3 + 1] = base_mass + neutron
            isotopes[i * 3 + 2] = base_mass + 2 * neutron

        return isotopes

    def calculate_ion_series_masses(self, sequence):

        """
        Calculate b and y ion series masses for a peptide sequence.
        """

        # Map each amino acid in the sequence to its corresponding mass
        seq_masses = np.array([amino_acid[aa] for aa in sequence])
        # Reverse the sequence masses for y ion calculation
        rev_masses = seq_masses[::-1]

        # Compute cumulative sums to calculate b and y ion series masses
        # Calculate b-series masses
        b_masses = np.cumsum(seq_masses)
        # Calculate y-series masses (reverse of b-series)
        y_masses = np.cumsum(rev_masses)

        # Calculate isotopic masses
        b_isotopes = self.calculate_isotopes(b_masses)
        y_isotopes = self.calculate_isotopes(y_masses)

        # Sort and return results
        return np.sort(b_isotopes), np.sort(y_isotopes)

    @staticmethod
    def calculate_residue_ions(b, y, charge, precursor_mass):
        """
        Generate b and y ion series with isotopic charge states.

        Parameters:
        - b: b-series masses (numpy array)
        - y: y-series masses (numpy array)
        - charge: maximum charge state to consider
        - precursor_mass: precursor mass

        Returns:
        - result: unique mass values (numpy array)
        """
        # Preallocate lists for b_ions and y_ions
        b_ions = []
        y_ions = []

        # Generate b-series ions
        for i in range(1, charge):
            b_ions.extend((b + (proton * i)) / i)
            b_ions.extend((b - h20 + (proton * i)) / i)
            b_ions.extend((b - nh3 + (proton * i)) / i)

        # Generate y-series ions
        for i in range(1, charge):
            y_ions.extend((y + h20 + (proton * i)) / i)
            y_ions.extend((y + (proton * i)) / i)  # h20 already included
            y_ions.extend((y - nh3 + h20 + (proton * i)) / i)

        # Include precursor mass
        y_ions.append(precursor_mass)

        # Combine and get unique results
        result = np.unique(np.concatenate([np.array(b_ions), np.array(y_ions)]))

        return result

    @staticmethod
    def remove_matching_peaks(experimental_peaks, theoretical_peaks):

        """
        Remove peaks from the experimental data that match theoretical peaks.

        Parameters:
        - experimental_peaks: observed experimental peaks (list)
        - theoretical_peaks: calculated theoretical peaks (numpy array)

        Returns:
        - filtered experimental peaks (numpy array)
        """

        # Collect indices of experimental peaks that match theoretical peaks
        arr = [
            idx for idx, val in enumerate(experimental_peaks)
            if any(i - fragment_tolerance <= val <= i + fragment_tolerance for i in theoretical_peaks)
        ]

        # Remove matching peaks based on the collected indices
        return np.delete(experimental_peaks, arr)

    @staticmethod
    def calculate_subpeptide_masses(peptide):

        """
        Calculate masses for all theoretical internal fragment ions
        """

        # Remove the first and last character of the peptide
        subpeptide_sequence = peptide[1:-1]

        # Generate all possible subpeptides of varying lengths
        subpeptide_set = {subpeptide_sequence[i:i + n]
                          for n in range(1, len(subpeptide_sequence) + 1)
                          for i in range(len(subpeptide_sequence) - n + 1)}

        # Calculate the mass of each subpeptide
        subpeptide_masses = {sum(amino_acid[aa] for aa in subpeptide)
                             for subpeptide in subpeptide_set}

        return np.array(sorted(subpeptide_masses))

    @staticmethod
    def consider_charge(fragment_ion_mass_list, charge):

        """
        Calculate ion masses for multiple charge states.

        Parameters:
        - fragment_ion_mass_list: list of base masses (numpy array)
        - charge: charge state

        Returns:
        - unique ion masses (numpy array)
        """

        # compute fragment ions masses for all charge states dynamically
        ion_masses = [(fragment_ion_mass_list + (proton * i)) / i for i in range(1, charge)]

        # Convert to a numpy array and remove duplicates
        return np.unique(np.array(ion_masses))

    @staticmethod
    def count_final_internal_fragments(filtered_experiment_peaks, theoretical_internal_fragments):

        """
        Count the number of matching internal fragment peaks.

        Parameters:
        - filtered_experiment_peaks: experimental peaks after filtering (numpy array)
        - theoretical_internal_fragments: theoretical internal fragment peaks (numpy array)

        Returns:
        - count of matching internal fragment peaks (integer)
        """

        # Identify a list of matching peaks within tolerance
        matching_indices = [
            idx for idx, peak in enumerate(filtered_experiment_peaks)
            if any(i - fragment_tolerance <= peak <= i + fragment_tolerance for i in theoretical_internal_fragments)
        ]

        # Return the number of matches
        return len(matching_indices)
