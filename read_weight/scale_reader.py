##################################################################
# Date: 10/06/2025
# Maintainer: Herve Mwunguzi
# Email: mwunguziher@gmail.com
# Purpose:
#   This class is for reading the output of an electronic scale 
#   using Serial (RS232).
#       Scale name: Precision Balance MK series
#       Baudrate: 9600  kbps
#       Output Mode: continuous output
##################################################################

import serial
from serial.serialutil import SerialException

class scale_reader:

    #scale reader constructor
    def __init__(self,port="/dev/ttyUSB0",baudrate=9600):

        self._serial = None
        #initializing the serial object
        try:
            self._serial = serial.Serial(port,baudrate,timeout=1)

        except FileNotFoundError:
            print(f"Error: Serial port '{port}' not found. Please check the port name.")

        except SerialException as e:

            if "Errno 16" in str(e) or "Resource busy" in str(e):
                print(f"Error: Serial port '{port}' is busy. Another program might be using it.")
                print("Please close any other applications that might be accessing the port.")
            elif "Permission denied" in str(e):
                print(f"Error: Permission denied for serial port '{port}'.")
                print("You might need to adjust user permissions or run with elevated privileges.")
            else:
                print(f"An unexpected SerialException occurred: {e}")

        except Exception as e:
            print(f"An unhandled error occurred: {e}")

    #read weight from the scale as it is sent
    def read_weight(self):
        if self._serial == None:
            print(f"Error happend while initializing serial communication.")
        else:
            #return the weight including the units
            return self._serial.readline().decode('utf-8').strip()
    
    #read only the value from the scale 
    def read_weight_as_value(self):
        if self._serial == None:
            print(f"Error happend while initializing serial communication.")
        else:
            #return the weight with no unit and as a float
            return float(self._serial.readline().decode('utf-8').rstrip('gctgnoz\r\n'))
    
    #desctructor
    def __del__(self):
        #close the port when out of scope
        if self._serial != None:
            self._serial.close()

    
if __name__ == '__main__':
    
    scale = scale_reader()

    while True:
        print(scale.read_weight_as_value())