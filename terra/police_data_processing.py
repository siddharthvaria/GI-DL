__author__ = 'robertk'
import sys
# All locations are a tuple of tuples of lat long of North Western Corner to South Eastern Corner
def filter_location(locations, data):
    new_data = []
    for line in data:
        inst = line.split(',')
        try:
            lat = float(inst[3])
            lon = float(inst[4])
#            lat = float(inst[19])
#            lon = float(inst[20])
        except (ValueError):
            lat = 0.0
            lon = 0.0
        for location in locations:
            if location[1][0] < lat < location[0][0] and location[0][1] < lon < location[1][1]:
                new_data.append(line)
    return new_data

# Splits a time string into its parts ordered from biggest to smallest
def split_time(time):
    datetime = []
    time_split = time.split()
    date_parts = time_split[0].split('/')
    time_parts = time_split[1].split(':')
    datetime.append(float(date_parts[2]))
    datetime.append(float(date_parts[0]))
    datetime.append(float(date_parts[1]))

    datetime.append(float(time_parts[0]))
    datetime.append(float(time_parts[1]))
    return datetime


# 0 if equal, -1 if test_time < control_time, 1 if test_time > control_time
def compare_time(test_time, control_time):
    test_datetime = split_time(test_time)
    control_datetime = split_time(control_time)
    for i in range(len(test_datetime)):
        if test_datetime[i] > control_datetime[i]:
            return 1
        elif test_datetime[i] < control_datetime[i]:
            return -1
    return 0


def filter_date(dates, data):
    new_data = []
    for line in data:
        inst = line.split(',')
        test_time = inst[0]
        for date in dates:
            cmp1 = compare_time(test_time, date[0])
            cmp2 = compare_time(test_time, date[1])
            if cmp1 > 0 and cmp2 < 0:
                new_data.append(line)
    return new_data

def filter_categories(categories, data):
    new_data = []
    for line in data:
        inst = line.split(',')
        test_category = inst[5]
        for category in categories:
            if test_category == category:
                new_data.append(line)
    return new_data

def main():
    file_path = '../data/chicago_police/chicago_crime_firearm.csv'
    #file_path = '../data/chicago_police/test.csv'
    #file_path = '../data/chicago_police/Crimes_-_2001_to_present.csv'
    data_file = open(file_path)
    line = data_file.readline()
    header = line.split(',')
    data = data_file.readlines()

    # Northwestern corner to southeastern corner
    # ((Lat, Long), (Lat, Long))
    locations = [
        ((41.783841, -87.614370), (41.776787, -87.609319)),
        ((41.779991, -87.618097), (41.774582, -87.614932)),
        ((41.783483, -87.640084), (41.776193, -87.634035))
    ]
    # ('D/M/YYYY H:MM', 'D/M/YYYY H:MM')
    dates = [
        ('1/1/14 00:00', '5/1/14 00:00')
    ]
    categories = [
        'HOMICIDE',
        'ASSAULT',
        'BATTERY',
    #    'WEAPONS VIOLATION',
    #    'CRIMINAL DAMAGE',
    #    'PUBLIC PEACE VIOLATION',
    ]
    fbi_codes = [
        '01A',  # Homicide
        '08A',  # Assault Simple
        '04A',  # Assault Aggrivated
        '2',  # Sexual Assault
        '08B',  # Battery Simple
        '04B',  # Battery Aggrivated
        '14',  # Criminal Damage
        '15',  # Weapons Violation
        '26',  # Other Weapons Violation
        '24'  # Public Peace Violation
    ]
    data = filter_location(locations, data)
    data = filter_date(dates, data)
    #data = filter_categories(categories, data)
    data.insert(0, line)
    for line in data:
        sys.stdout.write(line)

# query: https://data.cityofchicago.org/resource/6zsd-86xi.csv?$where=
# Location Filters
# ((latitude<41.783841 AND latitude>41.776787 AND longitude>-87.614370 AND longitude<-87.609319) OR
# (latitude<41.779991 AND latitude>41.774582 AND longitude>-87.618097 AND longitude<-87.614932) OR
# (latitude<41.783483 AND latitude>41.776193 AND longitude>-87.640084 AND longitude<-87.634035))
# Date Filter:
# (date>'2014-01-01T00:00:00.000' AND date<'2014-05-01T00:00:00.000')
# Category Filter:
# (fbi_code='01A' OR fbi_code='08A' OR fbi_code='04A' OR fbi_code='2' OR fbi_code='08B' OR fbi_code='04B' OR fbi_code='14' OR fbi_code='15' OR fbi_code='26' OR fbi_code='24')

#Full: https://data.cityofchicago.org/resource/6zsd-86xi.csv?$where=(((latitude<41.783841 AND latitude>41.776787 AND longitude>-87.614370 AND longitude<-87.609319) OR (latitude<41.779991 AND latitude>41.774582 AND longitude>-87.618097 AND longitude<-87.614932) OR (latitude<41.783483 AND latitude>41.776193 AND longitude>-87.640084 AND longitude<-87.634035)) AND (date>'2014-01-01T00:00:00.000' AND date<'2014-05-01T00:00:00.000') AND (fbi_code='01A' OR fbi_code='08A' OR fbi_code='04A' OR fbi_code='2' OR fbi_code='08B' OR fbi_code='04B' OR fbi_code='14' OR fbi_code='15' OR fbi_code='26' OR fbi_code='24'))
if __name__ == "__main__":
    main()
