import os
import logging
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.cluster import DBSCAN
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def top1(dataset) -> pd.DataFrame:
    """
    This ensures only the top-ranked result among the top N candidates is retained.

    Args:
        dataset (pd.DataFrame): Input dataset containing ranked results.

    Returns:
        pd.DataFrame: Dataset containing only the top 1 ranked result.
    """

    return dataset.drop_duplicates(subset=['Source File', 'Scan number'], keep='first').reset_index(drop=True)


def drop_missing_cluster_info(dataset) -> pd.DataFrame:
    """
    Removes rows from the dataset where clustering information is missing.

    Args:
        dataset (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Dataset with rows containing clustering information.
    """

    return dataset.dropna(subset=['Cluster']).reset_index(drop=True)


def labeling(dataset) -> pd.DataFrame:
    """
    Adds a 'Label' column to the dataset by comparing the 'Sequence' and 'GT' (Ground Truth) columns.

    Args:
        dataset (pd.DataFrame): A DataFrame containing 'Sequence' and 'GT' columns.

    Returns:
        pd.DataFrame: The original DataFrame with an added 'Label' column.

    Labeling logic:
        - 0 if 'Sequence' and 'GT' are the same (match)
        - 1 if 'Sequence' and 'GT' are different (mismatch)
    """

    dataset['Label'] = (dataset['Sequence'] != dataset['GT']).astype(int)

    return dataset


class ClusterRefinement:
    def __init__(self, cluster_info, mgf_info):

        """
        Initialize the ClusterRefinement class with a dataset.
        Merges information and cluster data into a single dataset.
        """

        self.dataset = pd.merge(mgf_info, cluster_info, how='left')

    def combine_cluster_and_charge(self):

        """
        Combine the cluster number and charge to create unique cluster IDs.
        """

        return [f"{cluster}_{int(charge)}"
                for cluster, charge in self.dataset[['Cluster', 'Charge']].values]

    @staticmethod
    def convert_to_integer_id(combined_clusters):

        """
        Map the combined cluster IDs to unique integer values.
        """

        # Create a dictionary to map unique cluster IDs to integers
        unique_clusters = {cluster_id: idx for idx, cluster_id in enumerate(sorted(set(combined_clusters)))}

        # Map each cluster ID to its corresponding integer
        return [unique_clusters[cluster_id] for cluster_id in combined_clusters]

    def refine_clusters_with_charge(self):

        """
        Refine the dataset by redefining cluster indices based on charge.
        """

        logger.info("Refining clusters with charge...")
        try:
            # Combine cluster number with charge
            combined_clusters = self.combine_cluster_and_charge()

            # Map combined cluster numbers to integers
            self.dataset['Cluster'] = self.convert_to_integer_id(combined_clusters)

            logger.info("Clusters refined with charge successfully.")
        except Exception as e:
            logger.error(f"Error refining clusters with charge: {e}")
            raise

    @staticmethod
    def refine_clusters_with_ppm(top_1, col, eps, min_samples):

        """
        Refine clusters based on Mass/Charge (m/z) and ppm.

        :param top_1:
        :param col: Column name to be used for refinement
        :param eps: DBSCAN parameter for the maximum distance between samples
        :param min_samples: DBSCAN parameter for the minimum number of samples in a cluster
        :return: DataFrame with new refined cluster labels
        """

        logger.info(f"Refining clusters with ppm. eps={eps}...")

        try:
            feature = top_1[[col, 'Mass/Charge (m/z)']]
            model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
            top_1['New Cluster'] = model.fit_predict(feature)

            logger.info("Clusters refined with ppm successfully.")

            return top_1
        except Exception as e:
            logger.error(f"Error refining clusters with ppm: {e}")
            raise

    @staticmethod
    def refine_clusters_with_rt(top_1, col, eps, min_samples):

        """
        Refine clusters based on retention time (RT).

        :param top_1:
        :param col: Column name to be used for refinement
        :param eps: DBSCAN parameter for the maximum distance between samples
        :param min_samples: DBSCAN parameter for the minimum number of samples in a cluster
        :return: DataFrame with new refined cluster labels
        """

        logger.info(f"Refining clusters with RT. eps={eps}...")

        try:
            feature = top_1[[col, 'Retention Time (min)']].copy()
            feature[col] = feature[[col]] * (eps + 1)  # Adjusting column by scaling with eps
            model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
            top_1['New Cluster'] = model.fit_predict(feature)

            logger.info("Clusters refined with RT successfully.")

            return top_1
        except Exception as e:
            logger.error(f"Error refining clusters with RT: {e}")
            raise

    @staticmethod
    def calculate_cluster_sizes(top_1, col):

        """
        Calculate the size of each cluster and add it as a new column.
        """

        logger.info("Calculating cluster sizes...")

        clu_list = top_1[col]
        clu_count = clu_list.value_counts().to_dict()
        top_1['New Count'] = top_1[col].map(clu_count)

        logger.info("Cluster sizes calculated successfully.")

        return top_1

    def merge_cluster_info(self, top_1):

        """
        Merge the refined cluster information into the dataset.
        """

        self.dataset = pd.merge(self.dataset, top_1[['Source File', 'Scan number', 'New Cluster', 'New Count']],
                                on=['Source File', 'Scan number'], how='left')

    def ml_cluster_refinement(self, initial_cluster_column, ppm_eps, rt_ppm, min_samples):

        """
        Perform machine learning-based refinement of clusters.
        """

        logger.info("Starting ML-based cluster refinement...")
        try:
            top_1 = top1(self.dataset)
            top_1 = self.refine_clusters_with_ppm(top_1, initial_cluster_column, ppm_eps, min_samples)
            top_1 = self.refine_clusters_with_rt(top_1, 'New Cluster', rt_ppm, min_samples)
            top_1 = self.calculate_cluster_sizes(top_1, 'New Cluster')
            self.merge_cluster_info(top_1)

            logger.info("ML-based cluster refinement completed successfully.")

        except Exception as e:
            logger.error(f"Error during ML-based cluster refinement: {e}")
            raise

    def execute(self, search_ppm, rt):

        """
        Execute the cluster refinement process with given search_ppm and RT parameters.

        :param rt:
        :param search_ppm: m/z tolerance
        """

        logger.info(f"Executing cluster refinement with search_ppm={search_ppm}, RT={rt}...")

        try:
            temp = self.dataset['Mass/Charge (m/z)'] * 1e-6 * search_ppm
            ppm_median = np.median(temp)

            self.refine_clusters_with_charge()  # Refine clusters based on charge
            self.ml_cluster_refinement('Cluster', ppm_median, rt, 1)  # Apply ML-based refinement

            logger.info("Cluster refinement executed successfully.")

        except Exception as e:
            logger.error(f"Error during execution: {e}")
            raise

    def get_refined_dataset(self):

        """
        Return the refined dataset with updated cluster information.
        """

        return self.dataset


class NewCandidatePeptideGenerator:
    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset

    def remove_missing_values(self):

        """
        Removes rows with missing values in 'Peptide' or 'Score' columns.

        Returns:
            pd.DataFrame: A DataFrame with missing values removed.
        """

        return self.dataset.dropna(subset=['Peptide', 'Score'])

    def accumulate_scores(self):

        """
        Accumulates scores for each peptide within each cluster.
        """

        score_dict = {}

        for cluster_id, peptide, score in self.dataset[['New Cluster', 'Peptide', 'Score']].values:

            if cluster_id not in score_dict:
                score_dict[cluster_id] = {}

            if peptide not in score_dict[cluster_id]:
                score_dict[cluster_id][peptide] = 0
            score_dict[cluster_id][peptide] += score * 0.01

        # Previous version (commented out): Scores were stored in lists and summed later
        #     if peptide not in score_dict[cluster_id]:
        #         score_dict[cluster_id][peptide] = []
        #     score_dict[cluster_id][peptide].append(score * 0.01)
        #
        # for cluster_id in score_dict:
        #     for peptide in score_dict[cluster_id]:
        #         score_dict[cluster_id][peptide] = sum(score_dict[cluster_id][peptide])

        return score_dict

    def select_top_n_candidates(self, top_n: int):

        """
        Selects the top n peptides based on scores for each cluster.

        Args:
            top_n (int): Number of top peptides to select per cluster.

        Returns:
            defaultdict: {0: [((ABCD, 6.2345), 0),((ACBD, 5.64321), 1)]}
                        A dictionary where each cluster ID maps to a list of
                        tuples containing (peptide, score) and rank.
        """

        logging.info(f"Selecting top {top_n} candidates for each cluster...")

        self.remove_missing_values()
        score_dict = self.accumulate_scores()

        # Initializes a defaultdict to avoid manual key initialization
        top_n_peptides = defaultdict(list)

        for cluster_id in score_dict:
            # Sort peptides by score in descending order and take the top n
            sorted_peptides = sorted(score_dict[cluster_id].items(), key=lambda x: x[1], reverse=True)[:top_n]

            # Assign a rank starting from 0
            for rank, peptide_and_score in enumerate(sorted_peptides):
                top_n_peptides[cluster_id].append((peptide_and_score, rank))

        logging.info(f"Top {top_n} candidate selection completed successfully.")

        return top_n_peptides

    def organize_new_candidates_info(self, top_peptides):

        """
        Organizes new candidate information by cluster, including peptides, sequences, ranks, and scores.

        Args:
            top_peptides (defaultdict): Top peptides with scores and ranks for each cluster.
       """

        logging.info("Organizing new candidate information by cluster...")

        peptides_by_cluster, sequences_by_cluster, ranks_by_cluster, scores_by_cluster = defaultdict(list), \
                                                                                         defaultdict(list), defaultdict(
            list), defaultdict(list)

        for peptide, sequence, cluster_id in self.dataset[['Peptide', 'Sequence', 'New Cluster']].values:

            for ranked_results in top_peptides[cluster_id]:
                if peptide == ranked_results[0][0]:
                    if peptide not in peptides_by_cluster[cluster_id]:
                        peptides_by_cluster[cluster_id].append(peptide)
                        sequences_by_cluster[cluster_id].append(sequence)
                        ranks_by_cluster[cluster_id].append(ranked_results[1])
                        scores_by_cluster[cluster_id].append(ranked_results[0][1])

        logging.info("Successfully organized new candidate information.")

        return peptides_by_cluster, sequences_by_cluster, ranks_by_cluster, scores_by_cluster

    @staticmethod
    def generate_new_candidates_dataframe(top_1_dataset, peptides_by_cluster, sequences_by_cluster, ranks_by_cluster,
                                          scores_by_cluster):

        """
        Generates a DataFrame of new candidate data.

        Args:
            top_1_dataset (pd.DataFrame): DataFrame containing the top 1 peptide data.

        Returns:
            pd.DataFrame: A DataFrame containing the new candidate peptides with their source file,
                          scan number, peptide, sequence, score, and rank.
        """

        logging.info("Generating new candidates DataFrame...")

        candidate_data = [
            (file_name, scan_number, peptide, sequence, score, rank)

            for file_name, scan_number, cluster_id in
            top_1_dataset[['Source File', 'Scan number', 'New Cluster']].values

            for peptide, sequence, rank, score in
            zip(peptides_by_cluster[cluster_id], sequences_by_cluster[cluster_id],
                ranks_by_cluster[cluster_id], scores_by_cluster[cluster_id])]

        logging.info(f"New candidates DataFrame generated successfully. Total rows: {len(candidate_data)}")

        return pd.DataFrame(candidate_data,
                            columns=['Source File', 'Scan number', 'Peptide', 'Sequence', 'Score', 'Rank'])


class MGFNoiseRemover:
    def __init__(self, mgf_directory, save_directory):

        """
        Initialize the MGFNoiseRemover with the directories for MGF files and the save location.

        :param mgf_directory: Path to the directory containing the MGF files.
        :param save_directory: Path to the directory where the cleaned MGF files will be saved.
        """

        self.mgf_directory = mgf_directory
        self.save_directory = save_directory

    def generate_noise_removed_mgf(self):

        """
        스팩트럼의 노이즈 픽을 제거하고 노이즈를 제거한 새로운 스팩트럼 파일(.mgf)을 저장하는 함수
        """

        mgf_list = os.listdir(self.mgf_directory)

        for file_name in tqdm(mgf_list):
            self.process_mgf_file(file_name)

    def process_mgf_file(self, filename):

        """
        하나의 MGF 파일을 읽고, 노이즈 제거 후 새로운 파일(_remove.mgf)로 저장
        """

        input_path = os.path.join(self.mgf_directory, filename)
        output_path = os.path.join(self.save_directory, f"{filename[:-4]}_remove.mgf")

        with open(input_path, 'r') as infile:
            data = infile.readlines()

        with open(output_path, 'w') as outfile:
            self.process_spectra(data, outfile)

    def process_spectra(self, data, outfile):

        """
        스펙트럼 데이터에서 노이즈를 제거하고 출력 파일에 기록
        """

        peak_list = []
        window_min, window_max = 0, 100

        for line in data:
            # BEGIN IONS 로 시작하는 줄
            if line.startswith('BEGIN IONS'):
                peak_list = []
                window_min, window_max = 0, 100
                outfile.write(line)
            # END IONS 로 시작하는 줄
            elif line.startswith('END IONS'):
                self.process_peaks(peak_list, outfile)
                outfile.write(line)
            else:
                # 숫자로 시작하는 줄
                if line.strip() and line.split()[0].replace('.', '', 1).isdigit():
                    mz = float(line.split()[0])
                    if mz <= window_max:
                        peak_list.append(line)
                    else:
                        window_min, window_max = self.adjust_window(mz, window_min, window_max)

                        if len(peak_list) == 0:
                            peak_list = [line]
                            continue

                        self.process_peaks(peak_list, outfile)
                        # 초기화
                        peak_list = [line]

                # 숫자로 시작하지 않는 줄
                else:
                    outfile.write(line)

    def process_peaks(self, peak_list, outfile):
        """END IONS를 만났을 때 temp 리스트의 피크를 처리하여 outfile에 기록"""
        # peak_list가 10개 이상의 피크를 가진 경우 노이즈 제거
        if len(peak_list) > 10:
            peak_list = self.remove_noise_peak_keep_order(peak_list)

        # peak_list의 모든 피크를 outfile에 기록
        for peak in peak_list:
            outfile.write(peak)

    @staticmethod
    def remove_noise_peak_keep_order(peak_list):

        intensity_list = [float(line.strip().split()[1]) for line in peak_list]

        while len(peak_list) > 10:
            # 최소 intensity의 인덱스 찾기
            min_index = intensity_list.index(min(intensity_list))

            # 해당 인덱스의 항목 제거
            peak_list.pop(min_index)
            intensity_list.pop(min_index)

        return peak_list

    @staticmethod
    def adjust_window(mz, window_min, window_max):

        """
        mz 값에 맞게 window_min과 window_max를 조정하는 함수
        """

        while mz > window_max:
            window_min += 100
            window_max += 100

        return window_min, window_max
