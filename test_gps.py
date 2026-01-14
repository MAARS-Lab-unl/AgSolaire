from read_gps.gps_reader import GPSReader

gps = GPSReader()
fix = gps.get_fix()
gps.close()

print("GPS FIX:", fix)