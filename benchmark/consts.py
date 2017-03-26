
MODEL_DIR_BASE = '/dfs/scratch0/fabuzaid/simdex/models/'
TO_RUN = [
        ('lemp-paper/Netflix-noav-10/', (10, 480189, 17770)),
        ('lemp-paper/Netflix-50/', (50, 480189, 17770)),
        ('lemp-paper/Netflix-noav-50/', (50, 480189, 17770)),
        ('lemp-paper/Netflix-noav-100/', (100, 480189, 17770)),
        ('lemp-paper/IE-nmf-10/', (10, 7716114, 1322094)),
        ('lemp-paper/IE-svd-10/', (10, 7716114, 1322094)),
        ('lemp-paper/IE-nmf-50/', (50, 7716114, 1322094)),
        ('lemp-paper/IE-svd-50/', (50, 7716114, 1322094)),
        ('lemp-paper/IE-nmf-100/', (100, 7716114, 1322094)),
        ('lemp-paper/IE-svd-100/', (100, 7716114, 1322094)),
]
# model_dir and K need to be formatted in
OUTPUT_DIR_BASE = '/dfs/scratch0/fabuzaid/simdex/experiments/models-{model_dir}/K-{K}/'

