from pathos.helpers import mp
import os

# We want only physical cores, not virtual cores. This means that we'll
# use every _other_ CPU in a given NUM node range
NUM_VIRTUAL_CORES_PER_POOL = 14


# Since we want only physical cores, we grab every other cpu id in the range
# [cpu_id_offset, cpu_id_offset + num_cores)
def get_cpu_assignments(cpu_id_offsets, num_cores):
    return (','.join(
        str(v)
        for v in range(cpu_id_offsets[0], cpu_id_offsets[0] + num_cores, 2)
    ) + ',' + ','.join(
        str(v)
        for v in range(cpu_id_offsets[1], cpu_id_offsets[1] + num_cores, 2)))

    # From `lscpu` on raiders6:


#   CPU(s):                112
#   On-line CPU(s) list:   0-111
#   Thread(s) per core:    2
#   Core(s) per socket:    14
#   Socket(s):             4
#   NUMA node(s):          4
#   NUMA node0 CPU(s):     0-13,56-69
#   NUMA node1 CPU(s):     14-27,70-83
#   NUMA node2 CPU(s):     28-41,84-97
#   NUMA node3 CPU(s):     42-55,98-111

NUM_NUMA_NODES = 4
NUMA_CPU_ID_OFFSETS = [(0, 56), (14, 70), (28, 84), (42, 98)]


# When assigning cores to a given process, use NUMA_QUEUE
# to fetch in real-time an available NUMA node. This guarantees
# that we won't have any contention.
def get_numa_queue(num_jobs_per_numa_node=1):
    m = mp.Manager()
    queue = m.Queue(NUM_NUMA_NODES * num_jobs_per_numa_node)
    for cpu_id_offsets in NUMA_CPU_ID_OFFSETS:
        for i in range(num_jobs_per_numa_node):
            queue.put(
                get_cpu_assignments(cpu_id_offsets,
                                    NUM_VIRTUAL_CORES_PER_POOL))
    return queue


DATASET_DIR_BASE = '/dfs/scratch0/fabuzaid/simdex/datasets/'
NETFLIX_DATASET = (
    DATASET_DIR_BASE + 'netflix-prize-dataset/netflix_train.tsv',
    DATASET_DIR_BASE + 'netflix-prize-dataset/netflix_test.tsv')
YAHOO_KDD_DATASET = (
    DATASET_DIR_BASE + 'yahoo-kdd/yahoo_kdd_2011_train_parsed.txt',
    DATASET_DIR_BASE + 'yahoo-kdd/yahoo_kdd_2011_test_parsed.txt')
YAHOO_R2_DATASET = (DATASET_DIR_BASE + 'yahoo-r2/yahoo_R2_train_full.txt',
                    DATASET_DIR_BASE + 'yahoo-r2/yahoo_R2_test_full.txt')

MODEL_DIR_BASE = '%s/models-simdex' % os.getenv('HOME')
LEMP_NETFLIX_MODELS = [
    ('lemp-paper/Netflix-noav-10', (10, 480189, 17770, (8, ), 1),
     NETFLIX_DATASET),
    ('lemp-paper/Netflix-noav-50', (50, 480189, 17770, (8, ), 1),
     NETFLIX_DATASET),
    ('lemp-paper/Netflix-noav-100', (100, 480189, 17770, (8, ), 1),
     NETFLIX_DATASET),
]

LEMP_MODELS = LEMP_NETFLIX_MODELS + [
    ('lemp-paper/KDD-50', (51, 1000990, 624961, (8, ), 20), YAHOO_KDD_DATASET),
]

FEXIPRO_MODELS = [
    ('fexipro-paper/Netflix-50', (50, 480189, 17770, (8, ), 1),
     NETFLIX_DATASET),
    ('fexipro-paper/KDD-50', (50, 1000990, 624961, (8, ), 20),
     YAHOO_KDD_DATASET),
]

LASTFM_MODELS = [
    ('lastfm/lastfm-10-75-iters-reg-0.01', (10, 358868, 292385, (8, ), 1),
     NETFLIX_DATASET),
    ('lastfm/lastfm-25-75-iters-reg-0.01', (25, 358868, 292385, (8, ), 1),
     NETFLIX_DATASET),
    ('lastfm/lastfm-50-75-iters-reg-0.01', (50, 358868, 292385, (8, ), 1),
     NETFLIX_DATASET),
    ('lastfm/lastfm-100-75-iters-reg-0.01', (100, 358868, 292385, (8, ), 1),
     NETFLIX_DATASET),
]

DSGDPP_MODELS = [
    ('dsgdpp/Netflix-10-reg-0.05', (10, 480189, 17770, (256, 512), 1),
     NETFLIX_DATASET),
    ('dsgdpp/Netflix-10-reg-0.5', (10, 480189, 17770, (256, 512), 1),
     NETFLIX_DATASET),
    ('dsgdpp/Netflix-10-reg-1.0', (10, 480189, 17770, (256, 512), 1),
     NETFLIX_DATASET),
    ('dsgdpp/Netflix-10-reg-5.0', (10, 480189, 17770, (256, 512), 1),
     NETFLIX_DATASET),
    ('dsgdpp/Netflix-10-reg-10.0', (10, 480189, 17770, (256, 512), 1),
     NETFLIX_DATASET),
    ('dsgdpp/Netflix-25-reg-0.05', (25, 480189, 17770, (256, 512), 1),
     NETFLIX_DATASET),
    ('dsgdpp/Netflix-25-reg-0.5', (25, 480189, 17770, (256, 512), 1),
     NETFLIX_DATASET),
    ('dsgdpp/Netflix-25-reg-1.0', (25, 480189, 17770, (256, 512), 1),
     NETFLIX_DATASET),
    ('dsgdpp/Netflix-25-reg-5.0', (25, 480189, 17770, (256, 512), 1),
     NETFLIX_DATASET),
    ('dsgdpp/Netflix-25-reg-10.0', (25, 480189, 17770, (256, 512), 1),
     NETFLIX_DATASET),
    ('dsgdpp/Netflix-50-reg-0.05', (50, 480189, 17770, (256, 512), 1),
     NETFLIX_DATASET),
    ('dsgdpp/Netflix-50-reg-0.5', (50, 480189, 17770, (256, 512), 1),
     NETFLIX_DATASET),
    ('dsgdpp/Netflix-50-reg-1.0', (50, 480189, 17770, (256, 512), 1),
     NETFLIX_DATASET),
    ('dsgdpp/Netflix-50-reg-5.0', (50, 480189, 17770, (256, 512), 1),
     NETFLIX_DATASET),
    ('dsgdpp/Netflix-100-reg-0.05', (100, 480189, 17770, (256, 512), 1),
     NETFLIX_DATASET),
    ('dsgdpp/Netflix-100-reg-0.5', (100, 480189, 17770, (256, 512), 1),
     NETFLIX_DATASET),
    ('dsgdpp/Netflix-100-reg-1.0', (100, 480189, 17770, (256, 512), 1),
     NETFLIX_DATASET),
    ('dsgdpp/Netflix-100-reg-5.0', (100, 480189, 17770, (256, 512), 1),
     NETFLIX_DATASET),
    ('dsgdpp/Netflix-100-reg-10.0', (100, 480189, 17770, (256, 512), 1),
     NETFLIX_DATASET),
]

GLOVE_MODELS = [
    ('lemp-paper/Glove-200', (200, 100000, 1093514, (256, 512), 1), ''),
]

NOMAD_NETFLIX_MODELS = [
    ('nomad/Netflix-10-reg-0.0005', (10, 480189, 17770, (256, 512), 1),
     NETFLIX_DATASET),
    ('nomad/Netflix-10-reg-0.005', (10, 480189, 17770, (256, 512), 1),
     NETFLIX_DATASET),
    ('nomad/Netflix-10-reg-0.05', (10, 480189, 17770, (256, 512), 1),
     NETFLIX_DATASET),
    ('nomad/Netflix-10-reg-0.5', (10, 480189, 17770, (256, 512), 1),
     NETFLIX_DATASET),
    ('nomad/Netflix-25-reg-0.0005', (25, 480189, 17770, (256, 512), 1),
     NETFLIX_DATASET),
    ('nomad/Netflix-25-reg-0.005', (25, 480189, 17770, (256, 512), 1),
     NETFLIX_DATASET),
    ('nomad/Netflix-25-reg-0.05', (25, 480189, 17770, (256, 512), 1),
     NETFLIX_DATASET),
    ('nomad/Netflix-25-reg-0.5', (25, 480189, 17770, (256, 512), 1),
     NETFLIX_DATASET),
    ('nomad/Netflix-50-reg-0.0005', (50, 480189, 17770, (256, 512), 1),
     NETFLIX_DATASET),
    ('nomad/Netflix-50-reg-0.005', (50, 480189, 17770, (256, 512), 1),
     NETFLIX_DATASET),
    ('nomad/Netflix-50-reg-0.05', (50, 480189, 17770, (256, 512), 1),
     NETFLIX_DATASET),
    ('nomad/Netflix-50-reg-0.5', (50, 480189, 17770, (256, 512), 1),
     NETFLIX_DATASET),
    ('nomad/Netflix-100-reg-0.0005', (100, 480189, 17770, (256, 512), 1),
     NETFLIX_DATASET),
    ('nomad/Netflix-100-reg-0.005', (100, 480189, 17770, (256, 512), 1),
     NETFLIX_DATASET),
    ('nomad/Netflix-100-reg-0.05', (100, 480189, 17770, (256, 512), 1),
     NETFLIX_DATASET),
    ('nomad/Netflix-100-reg-0.5', (100, 480189, 17770, (256, 512), 1),
     NETFLIX_DATASET),
]

NOMAD_KDD_MODELS = [
    ('nomad/KDD-10-reg-1', (10, 1000990, 624961, (512, 1024), 20),
     YAHOO_KDD_DATASET),
    ('nomad/KDD-10-reg-0.1', (10, 1000990, 624961, (512, 1024), 20),
     YAHOO_KDD_DATASET),
    ('nomad/KDD-10-reg-0.01', (10, 1000990, 624961, (512, 1024), 20),
     YAHOO_KDD_DATASET),
    ('nomad/KDD-10-reg-0.001', (10, 1000990, 624961, (512, 1024), 20),
     YAHOO_KDD_DATASET),
    ('nomad/KDD-25-reg-1', (25, 1000990, 624961, (512, 1024), 20),
     YAHOO_KDD_DATASET),
    ('nomad/KDD-25-reg-0.1', (25, 1000990, 624961, (512, 1024), 20),
     YAHOO_KDD_DATASET),
    ('nomad/KDD-25-reg-0.01', (25, 1000990, 624961, (512, 1024), 20),
     YAHOO_KDD_DATASET),
    ('nomad/KDD-25-reg-0.001', (25, 1000990, 624961, (512, 1024), 20),
     YAHOO_KDD_DATASET),
    ('nomad/KDD-50-reg-1', (50, 1000990, 624961, (512, 1024), 20),
     YAHOO_KDD_DATASET),
    ('nomad/KDD-50-reg-0.1', (50, 1000990, 624961, (512, 1024), 20),
     YAHOO_KDD_DATASET),
    ('nomad/KDD-50-reg-0.01', (50, 1000990, 624961, (512, 1024), 20),
     YAHOO_KDD_DATASET),
    ('nomad/KDD-50-reg-0.001', (50, 1000990, 624961, (512, 1024), 20),
     YAHOO_KDD_DATASET),
    ('nomad/KDD-100-reg-1', (100, 1000990, 624961, (512, 1024), 20),
     YAHOO_KDD_DATASET),
    ('nomad/KDD-100-reg-0.1', (100, 1000990, 624961, (512, 1024), 20),
     YAHOO_KDD_DATASET),
    ('nomad/KDD-100-reg-0.01', (100, 1000990, 624961, (512, 1024), 20),
     YAHOO_KDD_DATASET),
    ('nomad/KDD-100-reg-0.001', (100, 1000990, 624961, (512, 1024), 20),
     YAHOO_KDD_DATASET),
]

NOMAD_R2_MODELS = [
    ('nomad/R2-10-reg-1', (10, 1823179, 136736, (1024, 2048), 1),
     YAHOO_R2_DATASET),
    ('nomad/R2-10-reg-0.1', (10, 1823179, 136736, (1024, 2048), 1),
     YAHOO_R2_DATASET),
    ('nomad/R2-10-reg-0.01', (10, 1823179, 136736, (1024, 2048), 1),
     YAHOO_R2_DATASET),
    ('nomad/R2-10-reg-0.001', (10, 1823179, 136736, (1024, 2048), 1),
     YAHOO_R2_DATASET),
    ('nomad/R2-10-reg-0.0001', (10, 1823179, 136736, (1024, 2048), 1),
     YAHOO_R2_DATASET),
    ('nomad/R2-25-reg-1', (25, 1823179, 136736, (1024, 2048), 1),
     YAHOO_R2_DATASET),
    ('nomad/R2-25-reg-0.1', (25, 1823179, 136736, (1024, 2048), 1),
     YAHOO_R2_DATASET),
    ('nomad/R2-25-reg-0.01', (25, 1823179, 136736, (1024, 2048), 1),
     YAHOO_R2_DATASET),
    ('nomad/R2-25-reg-0.001', (25, 1823179, 136736, (1024, 2048), 1),
     YAHOO_R2_DATASET),
    ('nomad/R2-25-reg-0.0001', (25, 1823179, 136736, (1024, 2048), 1),
     YAHOO_R2_DATASET),
    ('nomad/R2-25-reg-0.00001', (25, 1823179, 136736, (1024, 2048), 1),
     YAHOO_R2_DATASET),
    ('nomad/R2-25-reg-0.000001', (25, 1823179, 136736, (1024, 2048), 1),
     YAHOO_R2_DATASET),
    ('nomad/R2-25-reg-0.0000001', (25, 1823179, 136736, (1024, 2048), 1),
     YAHOO_R2_DATASET),
    ('nomad/R2-25-reg-0', (25, 1823179, 136736, (1024, 2048), 1),
     YAHOO_R2_DATASET),
    ('nomad/R2-50-reg-1', (50, 1823179, 136736, (1024, 2048), 1),
     YAHOO_R2_DATASET),
    ('nomad/R2-50-reg-0.1', (50, 1823179, 136736, (1024, 2048), 1),
     YAHOO_R2_DATASET),
    ('nomad/R2-50-reg-0.01', (50, 1823179, 136736, (1024, 2048), 1),
     YAHOO_R2_DATASET),
    ('nomad/R2-50-reg-0.001', (50, 1823179, 136736, (1024, 2048), 1),
     YAHOO_R2_DATASET),
    ('nomad/R2-50-reg-0.0001', (50, 1823179, 136736, (1024, 2048), 1),
     YAHOO_R2_DATASET),
    ('nomad/R2-50-reg-0.00001', (50, 1823179, 136736, (1024, 2048), 1),
     YAHOO_R2_DATASET),
    ('nomad/R2-50-reg-0.000001', (50, 1823179, 136736, (1024, 2048), 1),
     YAHOO_R2_DATASET),
    ('nomad/R2-50-reg-0.0000001', (50, 1823179, 136736, (1024, 2048), 1),
     YAHOO_R2_DATASET),
    ('nomad/R2-50-reg-0', (50, 1823179, 136736, (1024, 2048), 1),
     YAHOO_R2_DATASET),
    ('nomad/R2-100-reg-1', (100, 1823179, 136736, (1024, 2048), 1),
     YAHOO_R2_DATASET),
    ('nomad/R2-100-reg-0.1', (100, 1823179, 136736, (1024, 2048), 1),
     YAHOO_R2_DATASET),
    ('nomad/R2-100-reg-0.01', (100, 1823179, 136736, (1024, 2048), 1),
     YAHOO_R2_DATASET),
    ('nomad/R2-100-reg-0.001', (100, 1823179, 136736, (1024, 2048), 1),
     YAHOO_R2_DATASET),
    ('nomad/R2-100-reg-0.0001', (100, 1823179, 136736, (1024, 2048), 1),
     YAHOO_R2_DATASET),
    ('nomad/R2-100-reg-0.00001', (100, 1823179, 136736, (1024, 2048), 1),
     YAHOO_R2_DATASET),
    ('nomad/R2-100-reg-0.000001', (100, 1823179, 136736, (1024, 2048), 1),
     YAHOO_R2_DATASET),
    ('nomad/R2-100-reg-0.0000001', (100, 1823179, 136736, (1024, 2048), 1),
     YAHOO_R2_DATASET),
    ('nomad/R2-100-reg-0', (100, 1823179, 136736, (1024, 2048), 1),
     YAHOO_R2_DATASET),
]

NOMAD_MODELS = NOMAD_NETFLIX_MODELS + NOMAD_KDD_MODELS + NOMAD_R2_MODELS

GOLD_STANDARD_MODELS = LEMP_MODELS + [
    ('nomad/Netflix-10-reg-0.05', (10, 480189, 17770,
                                   (8, ), 1), NETFLIX_DATASET),
    ('nomad/Netflix-25-reg-0.05', (25, 480189, 17770,
                                   (8, ), 1), NETFLIX_DATASET),
    ('nomad/Netflix-50-reg-0.05', (50, 480189, 17770,
                                   (8, ), 1), NETFLIX_DATASET),
    ('nomad/Netflix-100-reg-0.05', (100, 480189, 17770,
                                    (8, ), 1), NETFLIX_DATASET),
    ('nomad/R2-10-reg-0.001', (10, 1823179, 136736,
                               (8, ), 1), YAHOO_R2_DATASET),
    ('nomad/R2-25-reg-0.001', (25, 1823179, 136736,
                               (8, ), 1), YAHOO_R2_DATASET),
    ('nomad/R2-50-reg-0.000001', (50, 1823179, 136736,
                                  (8, ), 1), YAHOO_R2_DATASET),
    ('nomad/R2-100-reg-0', (100, 1823179, 136736, (8, ), 1), YAHOO_R2_DATASET),
    ('nomad/KDD-10-reg-1', (10, 1000990, 624961,
                            (8, ), 20), YAHOO_KDD_DATASET),
    ('nomad/KDD-25-reg-0.001', (25, 1000990, 624961,
                                (8, ), 20), YAHOO_KDD_DATASET),
    ('nomad/KDD-50-reg-1', (50, 1000990, 624961,
                            (8, ), 20), YAHOO_KDD_DATASET),
    ('nomad/KDD-100-reg-1', (100, 1000990, 624961,
                             (8, ), 20), YAHOO_KDD_DATASET),
]

INTERPOLATION_MODELS = [
    ('interpolation/KDD-100-reg-1-0.1-items',
     (100, 1000990, 62496, (512, 1024)), YAHOO_KDD_DATASET),
    ('interpolation/KDD-100-reg-1-0.1-users', (100, 100099, 624961,
                                               (256, 512)), YAHOO_KDD_DATASET),
    ('interpolation/KDD-100-reg-1-0.2-items',
     (100, 1000990, 124992, (512, 1024)), YAHOO_KDD_DATASET),
    ('interpolation/KDD-100-reg-1-0.2-users', (100, 200198, 624961,
                                               (256, 512)), YAHOO_KDD_DATASET),
    ('interpolation/KDD-100-reg-1-0.3-items',
     (100, 1000990, 187488, (512, 1024)), YAHOO_KDD_DATASET),
    ('interpolation/KDD-100-reg-1-0.3-users', (100, 300297, 624961,
                                               (256, 512)), YAHOO_KDD_DATASET),
    ('interpolation/KDD-100-reg-1-0.4-items',
     (100, 1000990, 249984, (512, 1024)), YAHOO_KDD_DATASET),
    ('interpolation/KDD-100-reg-1-0.4-users', (100, 400396, 624961,
                                               (256, 512)), YAHOO_KDD_DATASET),
    ('interpolation/KDD-100-reg-1-0.5-items',
     (100, 1000990, 312480, (512, 1024)), YAHOO_KDD_DATASET),
    ('interpolation/KDD-100-reg-1-0.5-users', (100, 500495, 624961,
                                               (256, 512)), YAHOO_KDD_DATASET),
    ('interpolation/KDD-100-reg-1-0.6-items',
     (100, 1000990, 374976, (512, 1024)), YAHOO_KDD_DATASET),
    ('interpolation/KDD-100-reg-1-0.6-users',
     (100, 600594, 624961, (512, 1024)), YAHOO_KDD_DATASET),
    ('interpolation/KDD-100-reg-1-0.7-items',
     (100, 1000990, 437472, (512, 1024)), YAHOO_KDD_DATASET),
    ('interpolation/KDD-100-reg-1-0.7-users',
     (100, 700693, 624961, (512, 1024)), YAHOO_KDD_DATASET),
    ('interpolation/KDD-100-reg-1-0.8-items',
     (100, 1000990, 499968, (512, 1024)), YAHOO_KDD_DATASET),
    ('interpolation/KDD-100-reg-1-0.8-users',
     (100, 800792, 624961, (512, 1024)), YAHOO_KDD_DATASET),
    ('interpolation/KDD-100-reg-1-0.9-items',
     (100, 1000990, 562464, (512, 1024)), YAHOO_KDD_DATASET),
    ('interpolation/KDD-100-reg-1-0.9-users',
     (100, 900891, 624961, (512, 1024)), YAHOO_KDD_DATASET),
]

TO_RUN = []

OTHER_MODELS = [
    ('pb-new/Netflix-10', (10, 480189, 17770), NETFLIX_DATASET),
    ('pb-new/Netflix-10-iters-100', (10, 480189, 17770), NETFLIX_DATASET),
    ('pb-new/Netflix-25', (25, 480189, 17770, (256, 512)), NETFLIX_DATASET),
    ('pb-new/Netflix-50', (50, 480189, 17770), NETFLIX_DATASET),
    ('sigmod-deadline/Netflix-10', (10, 480189, 17770), NETFLIX_DATASET),
    ('sigmod-deadline/Netflix-25', (25, 480189, 17770, (256, 512))),
    NETFLIX_DATASET,
    ('sigmod-deadline/Netflix-50', (50, 480189, 17770), NETFLIX_DATASET),
    ('pb-new/kdd-10', (10, 1000990, 624961), YAHOO_KDD_DATASET),
    ('pb-new/kdd-25', (25, 1000990, 624961), YAHOO_KDD_DATASET),
    ('pb-new/kdd-50', (50, 1000990, 624961), YAHOO_KDD_DATASET),
    ('sigmod-deadline/kdd-10', (10, 1000990, 624961), YAHOO_KDD_DATASET),
    ('sigmod-deadline/kdd-25', (25, 1000990, 624961), YAHOO_KDD_DATASET),
    ('sigmod-deadline/kdd-50', (50, 1000990, 624961), YAHOO_KDD_DATASET),
    ('ramdisk/r2-10', (10, 1823179, 136736), YAHOO_R2_DATASET),
    ('ramdisk/r2-25', (25, 1823179, 136736), YAHOO_R2_DATASET),
    ('ramdisk/r2-50', (50, 1823179, 136736), YAHOO_R2_DATASET),
    ('lemp-paper/Netflix-50', (50, 480189, 17770, (8, ), 1), NETFLIX_DATASET),
    ('lemp-paper/IE-svd-10', (10, 7716114, 1322094),
     NETFLIX_DATASET),  # Dataset is garbage value for the IE-* models
    ('lemp-paper/IE-svd-50', (50, 7716114, 1322094), NETFLIX_DATASET),
    ('lemp-paper/IE-svd-100', (100, 7716114, 1322094), NETFLIX_DATASET),
    ('lemp-paper/IE-nmf-10', (10, 7716114, 1322094), NETFLIX_DATASET),
    ('lemp-paper/IE-nmf-50', (50, 7716114, 1322094), NETFLIX_DATASET),
    ('lemp-paper/IE-nmf-100', (100, 7716114, 1322094), NETFLIX_DATASET),
]
