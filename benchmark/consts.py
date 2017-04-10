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

# We'll use NUMA node 1 for Simdex, NUMA node 2 for Blocked MM, and NUMA node 3
# for Lemp


SIMDEX_CPU_ID_OFFSETS = (14, 70)  # Simdex has CPU IDs 14-27,70-83
BLOCKED_MM_CPU_ID_OFFSETS = (28, 84)  # Blocked MM has CPU IDs 28-41,84-97
LEMP_CPU_ID_OFFSETS = (42, 98)  # Lemp has CPU IDs 42-55,98-111

SIMDEX_CPU_IDS = get_cpu_assignments(SIMDEX_CPU_ID_OFFSETS,
                                     NUM_VIRTUAL_CORES_PER_POOL)
BLOCKED_MM_CPU_IDS = get_cpu_assignments(BLOCKED_MM_CPU_ID_OFFSETS,
                                         NUM_VIRTUAL_CORES_PER_POOL)
LEMP_CPU_IDS = get_cpu_assignments(LEMP_CPU_ID_OFFSETS,
                                   NUM_VIRTUAL_CORES_PER_POOL)

MODEL_DIR_BASE = '/lfs/raiders6/ssd/fabuzaid/models-simdex'

TO_RUN = [
    ('lemp-paper/Netflix-noav-10', (10, 480189, 17770)),
    ('pb-new/Netflix-10', (10, 480189, 17770)),
    ('sigmod-deadline/Netflix-10', (10, 480189, 17770)),
    ('pb-new/Netflix-25', (25, 480189, 17770)),
    ('sigmod-deadline/Netflix-25', (25, 480189, 17770)),
    ('lemp-paper/Netflix-50', (50, 480189, 17770)),
    ('lemp-paper/Netflix-noav-50', (50, 480189, 17770)),
    ('pb-new/Netflix-50', (50, 480189, 17770)),
    ('sigmod-deadline/Netflix-50', (50, 480189, 17770)),
    ('lemp-paper/Netflix-noav-100', (100, 480189, 17770)),
    #('pb-new/kdd-10', (10, 1000990, 624961)),
    #('sigmod-deadline/kdd-10', (10, 1000990, 624961)),
    #('pb-new/kdd-25', (25, 1000990, 624961)),
    #('sigmod-deadline/kdd-25', (25, 1000990, 624961)),
    #('pb-new/kdd-50', (50, 1000990, 624961)),
    #('sigmod-deadline/kdd-50', (50, 1000990, 624961)),

    #('lemp-paper/IE-svd-10', (10, 7716114, 1322094)),
    #('lemp-paper/IE-svd-50', (50, 7716114, 1322094)),
    #('lemp-paper/IE-svd-100', (100, 7716114, 1322094)),
    #('pb-new/Netflix-10-iters-100', (10, 480189, 17770)),
    #('lemp-paper/IE-nmf-10', (10, 7716114, 1322094)),
    #('lemp-paper/IE-nmf-50', (50, 7716114, 1322094)),
    #('lemp-paper/IE-nmf-100', (100, 7716114, 1322094)),
]
