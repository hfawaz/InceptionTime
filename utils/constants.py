UNIVARIATE_DATASET_NAMES = ['50words', 'Adiac', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'Car', 'CBF',
                            'ChlorineConcentration', 'CinC_ECG_torso', 'Coffee',
                            'Computers', 'Cricket_X', 'Cricket_Y', 'Cricket_Z', 'DiatomSizeReduction',
                            'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW',
                            'Earthquakes', 'ECG200', 'ECG5000', 'ECGFiveDays', 'ElectricDevices', 'FaceAll', 'FaceFour',
                            'FacesUCR', 'FISH', 'FordA', 'FordB', 'Gun_Point', 'Ham', 'HandOutlines',
                            'Haptics', 'Herring', 'InlineSkate', 'InsectWingbeatSound', 'ItalyPowerDemand',
                            'LargeKitchenAppliances', 'Lighting2', 'Lighting7', 'MALLAT', 'Meat', 'MedicalImages',
                            'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW',
                            'MoteStrain', 'NonInvasiveFatalECG_Thorax1', 'NonInvasiveFatalECG_Thorax2', 'OliveOil',
                            'OSULeaf', 'PhalangesOutlinesCorrect', 'Phoneme', 'Plane', 'ProximalPhalanxOutlineAgeGroup',
                            'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW', 'RefrigerationDevices',
                            'ScreenType', 'ShapeletSim', 'ShapesAll', 'SmallKitchenAppliances', 'SonyAIBORobotSurface',
                            'SonyAIBORobotSurfaceII', 'StarLightCurves', 'Strawberry', 'SwedishLeaf', 'Symbols',
                            'synthetic_control', 'ToeSegmentation1', 'ToeSegmentation2', 'Trace', 'TwoLeadECG',
                            'Two_Patterns', 'UWaveGestureLibraryAll', 'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y',
                            'uWaveGestureLibrary_Z', 'wafer', 'Wine', 'WordsSynonyms', 'Worms', 'WormsTwoClass', 'yoga']

UNIVARIATE_DATASET_NAMES = ['Meat', 'Coffee']

UNIVARIATE_ARCHIVE_NAMES = ['TSC', 'InlineSkateXPs', 'SITS']
UNIVARIATE_ARCHIVE_NAMES = ['TSC']

SITS_DATASETS = ['SatelliteFull_TRAIN_c301', 'SatelliteFull_TRAIN_c200', 'SatelliteFull_TRAIN_c451',
                 'SatelliteFull_TRAIN_c89', 'SatelliteFull_TRAIN_c677', 'SatelliteFull_TRAIN_c59',
                 'SatelliteFull_TRAIN_c133']

InlineSkateXPs_DATASETS = ['InlineSkate-32', 'InlineSkate-64', 'InlineSkate-128',
                           'InlineSkate-256', 'InlineSkate-512', 'InlineSkate-1024',
                           'InlineSkate-2048']

dataset_names_for_archive = {'TSC': UNIVARIATE_DATASET_NAMES,
                             'SITS': SITS_DATASETS,
                             'InlineSkateXPs': InlineSkateXPs_DATASETS}
