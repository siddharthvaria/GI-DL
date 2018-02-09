__author__ = 'robertk'
from sodapy import Socrata


# https://data.cityofchicago.org/resource/6zsd-86xi.csv?$where=(((latitude<41.783841 AND latitude>41.776787 AND longitude>-87.614370 AND longitude<-87.609319) OR (latitude<41.779991 AND latitude>41.774582 AND longitude>-87.618097 AND longitude<-87.614932) OR (latitude<41.783483 AND latitude>41.776193 AND longitude>-87.640084 AND longitude<-87.634035)) AND (date>'2014-01-01T00:00:00.000' AND date<'2014-05-01T00:00:00.000') AND (fbi_code='01A' OR fbi_code='08A' OR fbi_code='04A' OR fbi_code='2' OR fbi_code='08B' OR fbi_code='04B' OR fbi_code='14' OR fbi_code='15' OR fbi_code='26' OR fbi_code='24'))
# query: https://data.cityofchicago.org/resource/6zsd-86xi.csv?$where=
# Location Filters
# ((latitude<41.783841 AND latitude>41.776787 AND longitude>-87.614370 AND longitude<-87.609319) OR
# (latitude<41.779991 AND latitude>41.774582 AND longitude>-87.618097 AND longitude<-87.614932) OR
# (latitude<41.783483 AND latitude>41.776193 AND longitude>-87.640084 AND longitude<-87.634035))
# Date Filter:
# (date>'2014-01-01T00:00:00.000' AND date<'2014-05-01T00:00:00.000')
# Category Filter:
# (fbi_code='01A' OR fbi_code='08A' OR fbi_code='04A' OR fbi_code='2' OR fbi_code='08B' OR fbi_code='04B' OR fbi_code='14' OR fbi_code='15' OR fbi_code='26' OR fbi_code='24')

def get_data(locations=None, dates=None, types=None, descriptions=None):
    client = Socrata("data.cityofchicago.org", None)
    url = 'http://data.cityofchicago.org/resource/6zsd-86xi.' +'csv'+'?'
    if locations is None and dates is None and types is None and descriptions is None:
        results = client.get('6zsd-86xi')
    else:
        where = '('
        if locations is not None:
            location_query = '('
            location = locations[0]
            location_query+='(latitude<='+str(location[0][0])+' AND latitude>='+str(location[1][0])+' AND longitude>='+str(location[0][1])+' AND longitude<='+str(location[1][1])+')'
            for i in range(1, len(locations)):
                location = locations[i]
                location_query += ' OR '
                location_query+='(latitude<='+str(location[0][0])+' AND latitude>='+str(location[1][0])+' AND longitude>='+str(location[0][1])+' AND longitude<='+str(location[1][1])+')'
            location_query += ')'
            where += location_query

        if dates is not None:
            if locations is not None:
                where += ' AND '
            date_query = '('
            date = dates[0]
            date_query += '(date>=\''+date[0]+'\' AND date<=\''+date[1]+'\')'
            for i in range(1, len(dates)):
                date = dates[i]
                date_query += ' OR '
                date_query += '(date>=\''+date[0]+'\' AND date<=\''+date[1]+'\')'
            date_query += ')'
            where += date_query

        if types is not None:
            if locations is not None or dates is not None:
                where += ' AND '
            type_query = '('
            type = types[0]
            type_query+='primary_type=\''+type+'\''
            for i in range(1, len(types)):
                type = types[i]
                type_query+=' OR '
                type_query+='primary_type=\''+type+'\''
            type_query += ')'
            where += type_query

        if descriptions is not None:
            if locations is not None or dates is not None and types is not None:
                where += ' AND '
            description_query = '('
            description = descriptions[0]
            description_query+='description=\''+description+'\''
            for i in range(1, len(descriptions)):
                description = descriptions[i]
                description_query+=' OR '
                description_query+='description=\''+description+'\''
            description_query += ')'
            where += description_query

        where += ' AND domestic=False'
        where += ')'
        url += '$where='+where
        print url
        results = client.get('6zsd-86xi', where=where, order='date ASC')
    return results

def main():
    locations = [
        ((41.783841, -87.614370), (41.776787, -87.609319)),
        ((41.779991, -87.618097), (41.774582, -87.614932)),
        ((41.783483, -87.640084), (41.776193, -87.634035))
    ]
    # ('YYYY-MM-DDTHH:MM:SS.MMM')
    dates = [
        ('2011-01-01T00:00:00.000', '2016-07-17T00:00:00.000')
        #('2016-07-17T00:00:00.000', '2016-07-24T00:00:00.000')
    ]

    # Make it more exclusive to firearm violence

    types = [
        'HOMICIDE',
        'ASSAULT',
        'BATTERY',
        'WEAPONS VIOLATION',
        'CRIMINAL DAMAGE',
        'PUBLIC PEACE VIOLATION',
    ]
    descriptions = [
        'FIRST DEGREE MURDER',
        'UNLAWFUL POSS OF HANDGUN',
        'UNLAWFUL USE HANDGUN',
        'UNLAWFUL SALE HANDGUN',
        'UNLAWFUL SALE OTHER FIREARM',
        'UNLAWFUL POSS OTHER FIREARM'
        'UNLAWFUL USE/SALE AIR RIFLE',
        'RECKLESS FIREARM DISCHARGE',
        'AGGRAVATED: HANDGUN',
        'AGGRAVATED OTHER FIREARM',
        'AGGRAVATED PO: OTHER FIREARM',
        'ARMED: HANDGUN',
        'ARMED: OTHER FIREARM',
        'ARMED: OTHER FIREARM',
        'ATTEMPT: ARMED-OTHER FIREARM'
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
    results = get_data(locations=locations, dates=dates, descriptions=descriptions)
    #results = get_data(locations=locations, dates=dates, types=types)
    #results = get_data('csv', locations=locations, dates=dates)
    #results = get_data('csv', dates=dates)
    dotw = [0]*7
    import datetime
    print
    for result in results:
        print result['date']
    #    dt_split = result['date'].split('T')
    #    date_split = dt_split[0].split('-')
    #    time_split = dt_split[1].split(':')
    #    time_split.append(time_split[2].split('.')[1])
    #    time_split[2] = time_split[2].split('.')[0]
    #    dotw[datetime.datetime(int(date_split[0]), int(date_split[1]), int(date_split[2]), int(time_split[0]), int(time_split[1]), int(time_split[2]), int(time_split[3])).weekday()] += 1
    total = sum(dotw)
    #days = ['Monday', 'Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    #for i in range(len(dotw)):
    #    day = dotw[i]
    #    print(days[i]+': '+(str(round(100*float(day)/float(total), 2)))+'%')
    #print dotw
    #print total

main()