from src.loader.dataloader import *
from src.utils import config, process, utils
from src.features.FeatureProcessor import *

if __name__ == '__main__':
    args = config.parse_arguments()
    config_path = args.config
    config_data = config.load_config(config_path)
    print("Configuration loaded successfully.")

    logging.info("Starting dataset preparation process...")
    processor = ClusterResultProcessor(config_data['path']['cluster_path'], config_data['path']['mgf_path'])
    cluster_df = processor.process_clusters()

    processor = MGFProcessor(config_data['path']['mgf_path'])
    mgf_information_df = processor.extract_spectrum_info()

    """
    Main function. Executes the training or testing process based on the TRAIN flag.
    """

    search_ppm = args.search_ppm
    cluster_rt = args.cluster_rt

    # Process based on the TRAIN flag
    if config_data['params']['train']:
        logging.info("Starting training process...")
        instance = process.TrainProcess(config_data)
        dataset = instance.execute_data_processing(cluster_df, mgf_information_df, search_ppm, cluster_rt)
    else:
        logging.info("Starting testing process...")
        instance = process.TestProcess(config_data)
        dataset = instance.execute_data_processing(cluster_df, mgf_information_df, search_ppm, cluster_rt)

    logging.info("Dataset preparation process completed successfully.")

    logging.info("Generating new candidates...")
    # Generate new top-N candidate peptides
    generate_process = process.GenerateNewCandidateProcess(config_data, dataset)
    new_dataset, dataset_top1 = generate_process.execute(top_n=2)

    logging.info("Starting the post-processing to create the new candidate peptide dataset...")
    new_dataset = instance.execute_candidate_generation(new_dataset, dataset_top1)
    logging.info(f"New candidate peptide dataset creation completed successfully. Final dataset size: {len(new_dataset)} rows.")

    logging.info("Candidate generation process completed successfully.")

    logging.info('Starting feature extraction process.')
    feature_transformer = FeatureTransformer(new_dataset)
    transformed_dataset = feature_transformer.generate_features()
    logging.info('Feature extraction process completed successfully.')

    denoised_mgf = process.SpectrumNoiseRemoverProcess(config_data)
    denoised_mgf.prepare_directory()
    denoised_mgf.execute()

    # Save the intermediate DataFrame as a CSV file
    logging.info("Starting to save the new dataset as CSV...")
    new_dataset.to_csv(os.path.join(config_data['path']['save_path'], "dataset_top2.csv"), index=False)
    logging.info("New dataset successfully saved as dataset_top2.csv.")

    if config_data['params']['top_10']:
        pass


# -> input, 타입 지정
#
