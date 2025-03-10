def get_phase(day, month):
    """
    Determine phase based on day and month.
    
    For the first cycle (dates not between March 16 and September 15):
        - Phase 1: September 16 to October 31
        - Phase 2: November, December, and January
        - Phase 3: February to March 15
        
    For the second cycle (dates between March 16 and September 15):
        - Phase 1: March 16 to April 30
        - Phase 2: May, June, and July
        - Phase 3: August to September 15
    """
    # First determine which cycle the date falls into.
    # We'll assume that if the date is between March 16 and September 15, it's in the second cycle.
    if (month > 3 or (month == 3 and day >= 16)) and (month < 9 or (month == 9 and day <= 15)):
        # Second cycle
        if (month == 3 and day >= 16) or (month == 4):
            return 1
        elif month in [5, 6, 7]:
            return 2
        elif month == 8 or (month == 9 and day <= 15):
            return 3
    else:
        # First cycle: either the date is before March 16 or after September 15.
        # Note: This cycle spans from September 16 through March 15.
        if (month == 9 and day >= 16) or (month == 10):
            return 1
        elif month in [11, 12, 1]:
            return 2
        elif month == 2 or (month == 3 and day <= 15):
            return 3

    # Fallback (shouldn't happen if date is valid)
    return None

# Examples:
print(get_phase(16, 9))   # For September 16 → First cycle, Phase 1
print(get_phase(10, 9))   # For September 10 → First cycle, but falls in Phase 2 or 3 depending on your interpretation
print(get_phase(12, 12))   # For March 20 → Second cycle, Phase 1