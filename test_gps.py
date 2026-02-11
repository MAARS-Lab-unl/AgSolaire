from read_gps.gps_reader import GPSReader

gps = GPSReader()
fix = gps.get_fix(max_wait_sec=60)
gps.close()

print("GPS FIX:", fix)