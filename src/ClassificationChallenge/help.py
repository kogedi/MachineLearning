import csv
#import EvaluateOneMe.csv


# Open the CSV file for reading
with open('.\src\ClassificationChallenge\EvaluateOnMe.csv', 'r') as csvfile:
    # Create a CSV reader object
    csvreader = csv.reader(csvfile)
    data_without_first_column = []
    
    # Iterate through rows in the CSV file
    for row in csvreader:
        # Exclude the first column and print the rest of the data
        currentrow = row[1:] #without first element
        newrow = currentrow[:-2] #
        newrow.append(currentrow[-1])
        data_without_first_column.append(newrow[:])
        #print(data_without_first_column)
        
output_file_path = '.\src\ClassificationChallenge\EvaluateOnMe2.csv'

# Open a new CSV file for writing
with open(output_file_path, 'w', newline='') as csvfile:
    # Create a CSV writer object
    csvwriter = csv.writer(csvfile)
    
    # Write the modified data (without the first column) to the output CSV file
    csvwriter.writerows(data_without_first_column)

print(f"Data without the first column has been saved to '{output_file_path}'.")