import csv
    
                
def read_case(case_ID):
    with open("cases.csv","r") as csvfile:
            file = csv.DictReader(csvfile, delimiter=',')
            for row in file:
                if row['ID'] == case_ID:
                    return row
    print("⚠️  Warning - Case not found!")
    return {}
    

def read_disturbance(disturbance_name):
    with open("disturbances.csv","r") as csvfile:
            file = csv.DictReader(csvfile, delimiter=',')
            for row in file:
                if row['Name'] == disturbance_name:
                    return row
    print("⚠️  Warning - Disturbance not found!")
    return {}


def read_fluid(fluid_name):
    with open("fluids.csv","r") as csvfile:
            file = csv.DictReader(csvfile, delimiter=',')
            for row in file:
                if row['Name'] == fluid_name:
                    return row
    print("⚠️  Warning - Fluid not found!")
    return {}


def write_result(case_ID, test_type, dp):
    with open("results.csv","a", newline='') as csvfile:
            file = csv.writer(csvfile, delimiter=',')
            row = [case_ID, test_type, dp]
            file.writerow(row)
    
