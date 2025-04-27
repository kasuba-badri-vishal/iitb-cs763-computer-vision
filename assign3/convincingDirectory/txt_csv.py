import csv
import argparse


parser = argparse.ArgumentParser(prog="txt to csv images",
        description="Convert txt data to csv data",
        epilog="Thank you for using it!",
        fromfile_prefix_chars="@")

parser.add_argument("-t", "--txt", required=True, help="Path to txt file")
parser.add_argument("-c", "--csv", required=True, help="Path to output csv")


args = parser.parse_args()

output_file = args.txt
csv_file = args.csv

# Open the output file and read the lines
with open(output_file, "r") as f:
    lines = f.readlines()

# Create a CSV writer and write the header
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Objective", "Objective Gap", "Risk", "Risk Gap", "Num Planes", "Iter"])

    # Loop through each line and parse the values
    for line in lines:
        if line.startswith("objective:"):
            objective = float(line.split(":")[1].strip())
        elif line.startswith("objective gap:"):
            objective_gap = float(line.split(":")[1].strip())
        elif line.startswith("risk:"):
            risk = float(line.split(":")[1].strip())
        elif line.startswith("risk gap:"):
            risk_gap = float(line.split(":")[1].strip())
        elif line.startswith("num planes:"):
            num_planes = int(line.split(":")[1].strip())
        elif line.startswith("iter:"):
            iteration = int(line.split(":")[1].strip())

            # Write the values to the CSV file
            writer.writerow([objective, objective_gap, risk, risk_gap, num_planes, iteration])
