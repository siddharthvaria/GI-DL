import datetime
import time
from sodapy import Socrata

def get_cpd_data(today_date):
    locations = [
        ((41.783841, -87.614370), (41.776787, -87.609319)),
        ((41.779991, -87.618097), (41.774582, -87.614932)),
        ((41.783483, -87.640084), (41.776193, -87.634035))
    ]

    #dates to get past 30 days
    today = today_date
    thirty_days = today-datetime.timedelta(days=30)
    curr_time = time.time()
    dates = [
            (str(thirty_days)+'T00:00:00.000', str(today)+'T00:00:00.000')
    ]

    # Make it more exclusive to firearm violence
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

    results = pull_cpd_data(locations=locations, dates=dates, descriptions=descriptions)
    return results

def pull_cpd_data(locations=None, dates=None, types=None, descriptions=None):
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

        where += ')'
        url += '$where='+where
        print url
        results = client.get('6zsd-86xi', where=where, order='date ASC')
    return results
