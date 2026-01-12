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
import re
from serial.serialutil import SerialException
import serial.tools.list_ports

class scale_reader:

    #scale reader constructor
    def __init__(self,port=None,baudrate=9600):

        self._serial = None

        # port_name = "Dtech USB Serial Controller"
        device_names = ["Prolific PL2303GT USB Serial","Dtech USB Serial Controller", "USB-SERIAL CH340"]
            
        port = self.find_device(device_names)

        # print(f"port: {port}")

         
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
            try:
                self._serial.readline().decode('utf-8').strip()
            except Exception as e:
                return None
            
            return self._serial.readline().decode('utf-8').strip()
            

    # read only the value from the scale 
    # def read_weight_as_value(self):
    #     if self._serial == None:
    #         print(f"Error happend while initializing serial communication.")
    #     else:
    #         # return the weight with no unit and as a float
    #         try:
    #             line = self._serial.readline().decode('utf-8').strip()
    #             parts = line.split()
    #             float(parts[2])
    #         except Exception as e:
    #             return None

    #         line = self._serial.readline().decode('utf-8').strip()
    #         parts = line.split()
    #         return float(parts[2])

    def read_weight_as_value(self):
        if self._serial is None:
            print("Error happened while initializing serial communication.")
            return None


        for _ in range(2):
            try:
                line = self._serial.readline().decode('utf-8', errors='ignore').strip()
                match = re.search(r'[-+]?\d+(?:\.\d+)?', line)
                if match:
                    return float(match.group(0))
            except Exception:
                pass

        return None



 
    #autodetect the scale COM port:
    def find_device(self, device_list):
        ports = serial.tools.list_ports.comports()
        for port in ports:
            # desc = port.description.lower()
            for device_name in device_list:
                if device_name in port.description:
                    print(f"Found {device_name} at {port.device}")
                    return port.device
        print(" No matching device found.")
        return None
    
    #desctructor
    def __del__(self):
        #close the port when out of scope
        if self._serial != None:
            self._serial.close()

    
if __name__ == '__main__':
    
    scale = scale_reader()

    while True:
        print(scale.read_weight_as_value())
        # print(scale.read_weight())
        # pass

