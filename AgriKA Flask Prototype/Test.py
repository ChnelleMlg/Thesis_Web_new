def get_phase(day, month):
    """
    Determines the phase based on the given day and month,
    and whether the date is the end of the season.

    First cycle (September 16 - March 15):
        Phase 1: September 16 - November 15
        Phase 2: November 16 - January 15
        Phase 3: January 16 - March 15 

    Second cycle (March 16 - September 15):
        Phase 1: March 16 - May 15
        Phase 2: May 16 - July 15
        Phase 3: July 16 - September 15 

    Returns:
        (phase: int, is_season_end: bool)
    """
    print(f"📅 DEBUG: Evaluating phase for {month=}, {day=}")
    
    # Second cycle: March 16 – September 15
    if (month == 3 and day >= 16) or (4 <= month <= 9 and (month != 9 or day <= 15)):
        if (month == 3 and day >= 16) or month == 4 or (month == 5 and day <= 15):
            return 1, False
        elif (month == 5 and day >= 16) or month == 6 or (month == 7 and day <= 15):
            return 2, False
        elif (month == 7 and day >= 16) or month == 8 or (month == 9 and day <= 15):
            return 3, True

    else:
        # First cycle: September 16 – March 15
        if (month == 9 and day >= 16) or month == 10 or (month == 11 and day <= 15):
            return 1, False
        elif (month == 11 and day >= 16) or month == 12 or (month == 1 and day <= 15):
            return 2, False
        elif (month == 1 and day >= 16) or month == 2 or (month == 3 and day <= 15):
            return 3, True

    return None, False  


print(get_phase(17, 3))  # Sept 15
# → (3, True)  # End of Season 2