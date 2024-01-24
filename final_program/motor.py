import gpiod
import time

chip = gpiod.Chip('gpiochip4')

class Motor():
    def __init__(self, In1, In2):
        self.In1 = In1
        self.In2 = In2
        
        self.lineIn1 = chip.get_line(In1)
        self.lineIn2 = chip.get_line(In2)
        
        self.lineIn1.request(consumer="ENGINE", type=gpiod.LINE_REQ_DIR_OUT)
        self.lineIn2.request(consumer="ENGINE", type=gpiod.LINE_REQ_DIR_OUT)
		
		
    def motorForward(self):
        self.lineIn1.set_value(0)
        self.lineIn2.set_value(1)
        
    def motorBackward(self):
        self.lineIn1.set_value(1)
        self.lineIn2.set_value(0)

    def motorStop(self):
        self.lineIn1.set_value(0)
        self.lineIn2.set_value(0)


motorFrontLeft = Motor(24, 23)
motorBackLeft = Motor(17, 27)
motorFrontRight = Motor(20, 21)
motorBackRight = Motor(26, 19)

def stop():
	motorFrontLeft.motorStop()
	motorBackLeft.motorStop()
	motorFrontRight.motorStop()
	motorBackRight.motorStop()
	time.sleep(0.25)
	
def moveForward():
	motorFrontLeft.motorForward()
	motorBackLeft.motorForward()
	motorFrontRight.motorForward()
	motorBackRight.motorForward()
	time.sleep(0.25)
	stop()

def moveBackward():
	motorFrontLeft.motorBackward()
	motorBackLeft.motorBackward()
	motorFrontRight.motorBackward()
	motorBackRight.motorBackward()
	time.sleep(0.25)
	stop()

def rotate90d_cw():
	motorFrontLeft.motorForward()
	motorBackLeft.motorForward()
	motorFrontRight.motorBackward()
	motorBackRight.motorBackward()
	time.sleep(0.390)
	stop()
	
def rotate180d_cw():
	motorFrontLeft.motorForward()
	motorBackLeft.motorForward()
	motorFrontRight.motorBackward()
	motorBackRight.motorBackward()
	time.sleep(0.78)
	stop()
	
def rotate90d_ccw():
	motorFrontLeft.motorBackward()
	motorBackLeft.motorBackward()
	motorFrontRight.motorForward()
	motorBackRight.motorForward()
	time.sleep(0.390)
	stop()
	
def rotate180d_ccw():
	motorFrontLeft.motorBackward()
	motorBackLeft.motorBackward()
	motorFrontRight.motorForward()
	motorBackRight.motorForward()
	time.sleep(0.78)
	stop()

#rotate180d()
#moveForward()
