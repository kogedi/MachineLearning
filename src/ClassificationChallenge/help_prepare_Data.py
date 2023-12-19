import csv
#import EvaluateOneMe.csv

#### INPUT TrainOnMe

#### OUTPUT 
# ChalY
# ChalX

cut_12 = True
cut_4 = True
cut_last = True

# # Open the CSV file for reading
# with open('C:/Users/Konrad Dittrich/git/repos/MachineLearning/src/ClassificationChallenge/TrainOnMe.csv', 'r') as csvfile:
#     # Create a CSV reader object
#     csvreader = csv.reader(csvfile)
#     data_without_first_column = []
    
#     # Iterate through rows in the CSV file
#     for row in csvreader:
#         # Exclude the first column and print the rest of the data
#         currentrow = row[1] # just the label
#         #newrow = currentrow[:-2] #
        
#         #newrow.append(currentrow[:])
#         data_without_first_column.append(row[1])
#         #print(data_without_first_column)
        
# output_file_path = 'C:/Users/Konrad Dittrich/git/repos/MachineLearning/src/ClassificationChallenge/ChalY.csv'

# # Open a new CSV file for writing
# with open(output_file_path, 'w', newline='') as csvfile:
#     # Create a CSV writer object
#     csvwriter = csv.writer(csvfile)
    
#     # Write the modified data (without the first column) to the output CSV file
#     csvwriter.writerows(data_without_first_column)

# print(f"Data without the first column has been saved to '{output_file_path}'.")

# ###### For the X Data


# Open the CSV file for reading
with open('C:/Users/Konrad Dittrich/git/repos/MachineLearning/src/ClassificationChallenge/TrainOnMe.csv', 'r') as csvfile:
    # Create a CSV reader object
    csvreader = csv.reader(csvfile)
    data_without_first_column = []
    
    # Iterate through rows in the CSV file
    for row in csvreader:
        # Exclude the first column and print the rest of the data
        
        
        if cut_4 == True:
            newrow = []
            newrow = row[2:6] + row [7:-1]
        
            data_without_first_column.append(newrow[:])
        else:
            currentrow = row[1] # just the label
            data_without_first_column.append(row[2:])

        
output_file_path = 'C:/Users/Konrad Dittrich/git/repos/MachineLearning/src/ClassificationChallenge/ChalX.csv'

# Open a new CSV file for writing
with open(output_file_path, 'w', newline='') as csvfile:
    # Create a CSV writer object
    csvwriter = csv.writer(csvfile)
    
    # Write the modified data (without the first column) to the output CSV file
    csvwriter.writerows(data_without_first_column)

print(f"Data without the first column has been saved to '{output_file_path}'.")
