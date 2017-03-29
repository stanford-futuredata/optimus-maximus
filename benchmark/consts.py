MODEL_DIR_BASE = '/lfs/raiders6/ssd/fabuzaid/models-simdex'
TO_RUN = [
    ('pb-new/Netflix-10', (10, 480189, 17770)),
    ('pb-new/Netflix-10-iters-100', (10, 480189, 17770)),
    ('pb-new/Netflix-25', (25, 480189, 17770)),
    ('pb-new/Netflix-50', (50, 480189, 17770)),
    #('pb-new/kdd-10', (10, 1000990, 624961)),
    #('pb-new/kdd-25', (25, 1000990, 624961)),
    #('pb-new/kdd-50', (50, 1000990, 624961)),
    ('sigmod-deadline/Netflix-10', (10, 480189, 17770)),
    ('sigmod-deadline/Netflix-25', (25, 480189, 17770)),
    ('sigmod-deadline/Netflix-50', (50, 480189, 17770)),
    #('sigmod-deadline/kdd-10', (10, 1000990, 624961)),
    #('sigmod-deadline/kdd-25', (25, 1000990, 624961)),
    #('sigmod-deadline/kdd-50', (50, 1000990, 624961)),
    ('lemp-paper/Netflix-noav-10', (10, 480189, 17770)),
    ('lemp-paper/Netflix-50', (50, 480189, 17770)),
    ('lemp-paper/Netflix-noav-50', (50, 480189, 17770)),
    ('lemp-paper/Netflix-noav-100', (100, 480189, 17770)),
    ('lemp-paper/IE-nmf-10', (10, 7716114, 1322094)),
    ('lemp-paper/IE-svd-10', (10, 7716114, 1322094)),
    ('lemp-paper/IE-nmf-50', (50, 7716114, 1322094)),
    ('lemp-paper/IE-svd-50', (50, 7716114, 1322094)),
    ('lemp-paper/IE-nmf-100', (100, 7716114, 1322094)),
    ('lemp-paper/IE-svd-100', (100, 7716114, 1322094)),
]
# model_dir and K need to be formatted in
OUTPUT_DIR_BASE = '/dfs/scratch0/fabuzaid/simdex/experiments/models-{model_dir}/K-{K}'
