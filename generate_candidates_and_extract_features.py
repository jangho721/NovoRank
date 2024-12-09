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
    elution_time = args.elution_time

    # Process based on the TRAIN flag
    if config_data['params']['train']:
        # Initiating the training process
        logging.info("Starting training process...")
        instance = process.TrainProcess(config_data)
        dataset = instance.execute_data_processing(cluster_df, mgf_information_df, search_ppm, cluster_rt)
    else:
        # Initiating the testing process
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

    logging.info('Starting feature extraction process...')
    feature_transformer = FeatureTransformer(new_dataset)
    feature_extraction = featureExtractor(config_data, elution_time)

    transformed_dataset = feature_transformer.generate_features()

    # Processing spectra to extract internal fragment ion features
    logging.info("Starting the spectrum denoising process...")
    denoised_mgf = process.SpectrumNoiseRemoveProcess(config_data)
    denoised_mgf.prepare_directory()
    denoised_mgf.execute()
    logging.info("Spectrum denoising process completed successfully.")

    logging.info("Starting internal fragment ion feature extraction...")
    new_dataset = feature_extraction.internal_fragment_ion_features(transformed_dataset)
    logging.info("Internal fragment ion feature extraction completed successfully.")

    logging.info("Starting retention time feature extraction...")
    new_dataset = feature_extraction.calculate_retention_time_difference_features(new_dataset)
    logging.info("Retention time feature extraction completed successfully.")
    logging.info('Feature extraction process completed successfully.')

    # Processing spectra to generate MGF files for XCorr calculation
    logging.info("Starting MGF file generation for XCorr calculation...")
    new_dataset = instance.execute_xcorr_mgf_generation(new_dataset)
    logging.info("MGF file generation for XCorr calculation completed successfully.")

    if config_data['params']['train']:
        # Training process code here
        logging.info("Training process completed successfully.")
    else:
        # Testing process code here
        logging.info("Testing process completed successfully.")

    # Save the intermediate DataFrame as a CSV file
    logging.info("Starting to save the new dataset as CSV...")
    new_dataset.to_csv(os.path.join(config_data['path']['save_path'],
                                    config_data['path']['interim_report']), index=False)
    logging.info(f"New dataset successfully saved as {config_data['path']['interim_report']}.")

    if config_data['params']['top_10']:
        pass

    logging.info("Proceeding to the next step:"
                 "\n\n1. Calculate the XCorr using CometX software"
                 "\n2. Execute run_novorank.py")

# -> input: 타입 지정
