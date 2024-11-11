from machine import Pin, PWM
import sys
import time
import _thread

"""
This is the micropython code that runs on the pico controlling the ball machine
The robot communicates with the controller computer over usb serial
"""
class MotorController:
    def __init__(self):
        # Single step pin for all steppers
        self.step_pin = Pin(3, Pin.OUT)  # GP3

        # Direction pins for each stepper
        self.dir_pin1 = Pin(2, Pin.OUT)  # GP2
        self.dir_pin2 = Pin(4, Pin.OUT)  # GP4
        self.dir_pin3 = Pin(28, Pin.OUT)  # GP28
        self.dir_pin4 = Pin(8, Pin.OUT)  # GP8

        # PWM motor pins
        self.pwm1 = PWM(Pin(10))  # GP10
        self.pwm2 = PWM(Pin(11))  # GP11
        self.reverse_pin = Pin(12, Pin.OUT)  # GP12

        # Set PWM frequency
        self.pwm1.freq(500)
        self.pwm2.freq(500)

        # Initialize variables with lock for thread safety
        self.lock = _thread.allocate_lock()
        self.stepper_rate = 0
        self.pwm_value = 0
        self.motor_direction = 0

        # Start the motor control on the second core
        _thread.start_new_thread(self.motor_control_thread, ())

    def map_value(self, x, in_min, in_max, out_min, out_max):
        return int((x - in_min) * (out_max - out_min) // (in_max - in_min) + out_min)

    def process_steppers(self):
        with self.lock:
            current_rate = self.stepper_rate

        if current_rate != 0:
            direction = 1 if current_rate > 0 else 0
            self.dir_pin1.value(direction)
            self.dir_pin2.value(direction)
            self.dir_pin3.value(direction)
            self.dir_pin4.value(direction)

            step_delay_us = self.map_value(abs(current_rate), 1, 100, 10000, 500)

            self.step_pin.value(1)
            time.sleep_us(10)
            self.step_pin.value(0)
            time.sleep_us(step_delay_us)

    def set_pwm_motors(self):
        with self.lock:
            current_pwm = self.pwm_value
            current_dir = self.motor_direction

        pwm_scaled = self.map_value(current_pwm, 0, 100, 0, 65535)
        self.pwm1.duty_u16(pwm_scaled)
        self.pwm2.duty_u16(pwm_scaled)
        self.reverse_pin.value(current_dir)

    def parse_data(self, data):
        try:
            parts = data.strip().split(',')
            if len(parts) >= 3:
                with self.lock:
                    self.stepper_rate = int(parts[0])
                    self.pwm_value = int(parts[1])
                    self.motor_direction = int(parts[2])
                sys.stdout.write("ACK\n")
        except Exception as e:
            sys.stdout.write("ERR\n")

    def motor_control_thread(self):
        """Motor control running on core 1"""
        print("Motor control started on core 1")
        while True:
            self.process_steppers()
            time.sleep_ms(1)

    def run(self):
        """Serial handling on core 0 (main core)"""
        print("Serial handler running on core 0")
        while True:
            try:
                data = sys.stdin.readline()
                if data:
                    self.parse_data(data)
                    self.set_pwm_motors()
            except Exception as e:
                sys.stdout.write(str(e))
            time.sleep_ms(10)


def main():
    try:
        controller = MotorController()
        print("Motor controller initialized")
        controller.run()  # This runs on core 0
    except Exception as e:
        print(f"Fatal error: {e}")
        led = Pin(25, Pin.OUT)
        while True:
            led.toggle()
            time.sleep_ms(500)


if __name__ == "__main__":
    main()
