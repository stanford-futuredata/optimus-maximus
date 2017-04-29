NETFLIX_REGS = ['0.0005', '0.005', '0.05', 'gold-standard', '0.5']
KDD_REGS = ['0.001', '0.01', '0.1', 'gold-standard', '1']
R2_REGS = ['0.001', '0.01', '0.1', 'gold-standard', '1']

NETFLIX_10_MODELS = [
    'lemp-paper-Netflix-noav-10',
    #'pb-new-Netflix-10',
    #'sigmod-deadline-Netflix-10',
]
NETFLIX_25_MODELS = [
    #'pb-new-Netflix-25',
    #'sigmod-deadline-Netflix-25',
]
NETFLIX_50_MODELS = [
    'lemp-paper-Netflix-50',
    'lemp-paper-Netflix-noav-50',
    #'sigmod-deadline-Netflix-50',
]
NETFLIX_100_MODELS = [
    'lemp-paper-Netflix-noav-100',
]
NETFLIX_MODELS = NETFLIX_10_MODELS + NETFLIX_25_MODELS + NETFLIX_50_MODELS + NETFLIX_100_MODELS


KDD_10_MODELS = [
    'nomad-KDD-10-reg-0.001',
    'nomad-KDD-10-reg-0.01',
    'nomad-KDD-10-reg-0.1',
    'nomad-KDD-10-reg-1',
    #'pb-new-kdd-10',
    #'sigmod-deadline-kdd-10',
]
KDD_25_MODELS = [
    'nomad-KDD-25-reg-0.001',
    'nomad-KDD-25-reg-0.01',
    'nomad-KDD-25-reg-0.1',
    'nomad-KDD-25-reg-1',
    #'pb-new-kdd-25',
    #'sigmod-deadline-kdd-25',
]
KDD_50_MODELS = [
    'lemp-paper-KDD-50',
    'nomad-KDD-50-reg-0.1',
    'nomad-KDD-50-reg-1',
    #'pb-new-kdd-50',
    #'sigmod-deadline-kdd-50',
]
KDD_100_MODELS = [
    'nomad-KDD-100-gold-standard',
    'nomad-KDD-100-reg-0.001',
    'nomad-KDD-100-reg-0.01',
    'nomad-KDD-100-reg-0.1',
]
KDD_MODELS = KDD_10_MODELS + KDD_25_MODELS + KDD_50_MODELS + KDD_100_MODELS


R2_10_MODELS = [
    'nomad-R2-10-reg-0.001',
    'nomad-R2-10-reg-0.01',
    'nomad-R2-10-reg-0.1',
    'nomad-R2-10-reg-1',
]
R2_25_MODELS = [
    'nomad-R2-25-reg-0.001',
    'nomad-R2-25-reg-0.01',
    'nomad-R2-25-reg-0.1',
    'nomad-R2-25-reg-1',
]
R2_50_MODELS = [
    'nomad-R2-50-reg-0.001',
    'nomad-R2-50-reg-0.01',
    'nomad-R2-50-reg-0.1',
    'nomad-R2-50-reg-1',
]
R2_100_MODELS = [
    'nomad-R2-100-reg-0.001',
    'nomad-R2-100-reg-0.01',
    'nomad-R2-100-reg-0.1',
    'nomad-R2-100-reg-1',
]
R2_MODELS = R2_10_MODELS + R2_25_MODELS + R2_50_MODELS + R2_100_MODELS
