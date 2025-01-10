# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 09:51:31 2024

@author: gauthambekal93
"""

from datetime import datetime, timedelta

# Given number of seconds
seconds = 27032360  #Nov 09
seconds =31536000
seconds = 5173160

seconds= (23-7)*24*3600  # Jan 17

#seconds =  (115-7)*24*3600   #april 19
seconds = 3735680.0
#seconds =  36881960 #27032360 +  9849600
#Summer period (from June 21st till September 22nd). 

# Reference date (January 1st of a non-leap year)
reference_date = datetime(year=2023, month=1, day=1)

# Calculate the new date by adding the timedelta to the reference date
new_date = reference_date + timedelta(seconds=seconds)

print(f"The corresponding date and time is: {new_date}")