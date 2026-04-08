MAJOR_TISSUE_LIST_PATH = '/mnt/afan/scGEPOP/data/cellxgene/query_list.txt'
with open(MAJOR_TISSUE_LIST_PATH) as f:
    MAJOR_TISSUE_LIST = [line.rstrip('\n') for line in f]
VERSION = "2025-01-30"

#  build the value filter dict for each tissue
VALUE_FILTER = {
    ethnicity : f"suspension_type != 'na' and self_reported_ethnicity == '{ethnicity}'" for ethnicity in MAJOR_TISSUE_LIST
}


ENTHNICITY_CONVERT = {
    'African': 0,
    'Asian': 1, 
    'European': 2, 
    'Indigenous': 3
}
ENTHNICITY_DICT = {
    'African American': ['African', 0],
    'Ethiopian': ['African', 0],
    'Arab': ['Asian', 1],
    'Asian': ['Asian', 1],
    'Bangladeshi': ['Asian', 1],
    'Chinese': ['Asian', 1],
    'East Asian': ['Asian', 1],
    'Han Chinese': ['Asian', 1],
    'Indian': ['Asian', 1],
    'Iraqi': ['Asian', 1],
    'Japanese': ['Asian', 1],
    'Korean': ['Asian', 1],
    'Singaporean Chinese': ['Asian', 1],
    'Singaporean Indian': ['Asian', 1],
    'Singaporean Malay': ['Asian', 1],
    'South Asian': ['Asian', 1],
    'British': ['European', 2],
    'European American': ['European', 2],
    'Finnish': ['European', 2],
    'German': ['European', 2],
    'Irish': ['European', 2],
    'Jewish Israeli': ['European', 2],
    'American': ['Indigenous', 3],
    'Native American': ['Indigenous', 3],
    'Pacific Islander': ['Indigenous', 3],
}


if __name__ == "__main__":
    # print(VALUE_FILTER["others"])
    # print(MAJOR_TISSUE_LIST)
    print(VALUE_FILTER)
