NETFLIX_REGS = ['0.0005', '0.005', '0.05', '0.5']
KDD_REGS = ['0.001', '0.01', '0.1', '1']
R2_REGS = [
    '0',
    '0.0000001',
    '0.000001',
    '0.00001',
    '0.0001',
    '0.001',
    '0.01',
]

NETFLIX_10_MODELS = [
    'lemp-paper-Netflix-noav-10',
    'nomad-Netflix-10-reg-0.0005',
    'nomad-Netflix-10-reg-0.005',
    'nomad-Netflix-10-reg-0.05',
    'nomad-Netflix-10-reg-0.5',
]
NETFLIX_25_MODELS = [
    'nomad-Netflix-25-reg-0.0005',
    'nomad-Netflix-25-reg-0.005',
    'nomad-Netflix-25-reg-0.05',
    'nomad-Netflix-25-reg-0.5',
]
NETFLIX_50_MODELS = [
    'lemp-paper-Netflix-50',
    'lemp-paper-Netflix-noav-50',
    'nomad-Netflix-50-reg-0.0005',
    'nomad-Netflix-50-reg-0.005',
    'nomad-Netflix-50-reg-0.05',
    'nomad-Netflix-50-reg-0.5',
]
NETFLIX_100_MODELS = [
    'lemp-paper-Netflix-noav-100',
    'nomad-Netflix-100-reg-0.0005',
    'nomad-Netflix-100-reg-0.005',
    'nomad-Netflix-100-reg-0.05',
    'nomad-Netflix-100-reg-0.5',
]
NETFLIX_MODELS = NETFLIX_10_MODELS + NETFLIX_25_MODELS + NETFLIX_50_MODELS + NETFLIX_100_MODELS

LEMP_NETFLIX_MODELS = [
    'lemp-paper-Netflix-noav-10',
    'lemp-paper-Netflix-noav-50',
    'lemp-paper-Netflix-50',
    'lemp-paper-Netflix-noav-100',
]

KDD_10_MODELS = [
    'nomad-KDD-10-reg-0.001',
    'nomad-KDD-10-reg-0.01',
    'nomad-KDD-10-reg-0.1',
    'nomad-KDD-10-reg-1',
]
KDD_25_MODELS = [
    'nomad-KDD-25-reg-0.001',
    'nomad-KDD-25-reg-0.01',
    'nomad-KDD-25-reg-0.1',
    'nomad-KDD-25-reg-1',
]
KDD_50_MODELS = [
    'lemp-paper-KDD-50',
    'nomad-KDD-50-reg-0.1',
    'nomad-KDD-50-reg-1',
]
KDD_100_MODELS = [
    'nomad-KDD-100-reg-1',
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

GOLD_STANDARD_MODELS = [
    'lemp-paper-Netflix-noav-10',
    'lemp-paper-Netflix-noav-50',
    'lemp-paper-Netflix-noav-100',
    'lemp-paper-KDD-50',
    'nomad-Netflix-10-reg-0.05',
    'nomad-Netflix-25-reg-0.05',
    'nomad-Netflix-50-reg-0.05',
    'nomad-Netflix-100-reg-0.05',
    'nomad-R2-10-reg-0.001',
    'nomad-R2-25-reg-0.001',
    'nomad-R2-50-reg-0.000001',
    'nomad-R2-100-reg-0',
    'nomad-KDD-10-reg-1',
    'nomad-KDD-25-reg-0.001',
    'nomad-KDD-50-reg-1',
    'nomad-KDD-100-reg-1',
]

BLOG_POST_MODELS = [
    'nomad-Netflix-25-reg-0.05',
    'nomad-Netflix-50-reg-0.05',
    'nomad-R2-25-reg-0.001',
    'nomad-R2-50-reg-0.001',
    'nomad-KDD-25-reg-0.001',
    'nomad-KDD-50-reg-1',
]
