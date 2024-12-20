#ifndef __MSCLUSTERICLUDES_H__
#define __MSCLUSTERICLUDES_H__

/*! @file MsClusterIncludes.h
	A central location for definitions and constants used in MsCluster.
*/

#include "../Common/includes.h"
#include "../Common/auxfun.h"
#include "../PepNovo/PepNovo_includes.h"

const size_t NUM_PEAKS_FOR_HEURISTIC       = 4;	 /*! Number of peaks used for deciding if two spectra should be compared.
												     If they don't have at least one peak mass in common in the top
													 \c NUM_PEAKS_FOR_HEURISTIC in common there is no reason to compare them.
													 It is not recmonneded to go below 3, and there is no benefit in going above 6.*/
const size_t NUM_TOP_SIMILARITIES_TO_SAVE  = 25;   /// number of distance computations to remember between rounds
const size_t MAX_NUM_PEAKS_FOR_DISTANCE    = 40;   /// Maximal number of peaks used in distance computation
const float MIN_SIMILARITY_FOR_ACCURATE_COMPUTATION = 0.1; /// minimal similarity required for accurate similarity computation (which includes peak intensities)
const size_t ALLOWED_DIFF_IN_NUM_DISTANCE_PEAKS = 5; /// maximum size difference in distance peaks that is tolerated between two spectra
const float	MAX_SIMILARITY				   = 0.7; /// do not require clusters to have a larger similarity than this for joining
const unsigned int LARGE_CLUSTER_SIZE      = 20;  /// number of spectra for cluster to be considered "big" and rquire higher similarity to add too
const unsigned int MAX_CLUSTER_SIZE_FOR_UPDATE   = 255; /// adding spectra beyond this size does not change the peaks in the consensus spectrum.
const size_t	SIMILARITY_BATCH_SIZE	   = 800;	/// how many specra should be clustered in each sweep
const size_t	MAX_SIMILARITY_LIST_SIZE   = 20000000; /// maximal number of pairs to store in list of pairs for similarity
const mass_t MAJOR_MZ_INCREMENT_FOR_DAT	   = 25.0; /*! output and first pass dat split files according to this value
												   of Daltions*/

const unsigned int SIZES_FOR_REDOING_DISTANCE_PEAKS[]={2,3,4,5,9,16};
const size_t   NUM_SIZES_FOR_REDOING_DISTANCE_PEAKS = sizeof(SIZES_FOR_REDOING_DISTANCE_PEAKS)/sizeof(unsigned int);

											

const clusterIdx_t    MAX_CLUSTER_IDX    = numeric_limits<clusterIdx_t>::max();
const longInt8_t	  MAX_INT8			 = numeric_limits<longInt8_t>::max();

#endif



