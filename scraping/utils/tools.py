
from datetime import date, timedelta, datetime


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


def get_monday_sunday_of_week(year, week_number):
    # Find the first day of the week (Monday) for the given year and week number
    first_day = datetime.strptime(f'{year}-W{week_number}-1', "%Y-W%W-%w")

    # Calculate the Sunday of the same week by adding 6 days to the first day
    last_day = first_day + timedelta(days=6)

    return (first_day, last_day)

def week_range(start_year, end_year):
    week_list = []
    for year in range(start_year, end_year + 1):
        for week_number in range(1, 53):
            week_list.append(get_monday_sunday_of_week(year, week_number))
    return week_list