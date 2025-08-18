import csv

# Initialize the lists
list1 = []
list2 = []
check = '/sty_mfn'

# Read the CSV file
with open('ind_mask_cmmd_stats.csv', mode='r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        if len(row) >= 2:
            first_col = row[0]
            third_col = float(row[1])
            
            # Check if 'sty' is in the first column
            if check in first_col:
                # Append to the appropriate list
                if 'CosineT2Adv' in first_col:
                    list1.append(third_col)
                else:
                    list2.append(third_col)

# Print the results
print(f"For: {check}")
print(f"DAFR: {sum(list1)/len(list1)} with {len(list1)}")
print(f"SAS: {sum(list2)/len(list2)} with {len(list2)}")