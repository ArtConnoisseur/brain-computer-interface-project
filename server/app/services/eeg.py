"""
This is a ground up redesign of the script provided
by the team behind the PiEEG-16, aimed at modularising 
the task and making the API compatible with modern
Python standards and this backend application.
"""


import spidev
import gpiod

class EEGReader:
    # ADS1299 command constants
    WAKEUP  = 0x02
    RESET   = 0x06
    START   = 0x08
    STOP    = 0x0A
    RDATAC  = 0x10
    SDATAC  = 0x11

    # Register addresses
    CONFIG1 = 0x01
    CONFIG2 = 0x02
    CONFIG3 = 0x03

    CH1SET  = 0x05
    CH2SET  = 0x06
    CH3SET  = 0x07
    CH4SET  = 0x08
    CH5SET  = 0x09
    CH6SET  = 0x0A
    CH7SET  = 0x0B
    CH8SET  = 0x0C

    DATA_TEST  = 0x7FFFFF
    DATA_CHECK = 0xFFFFFF

    def __init__(self):
        self.cs_pin = 19

        # GPIO / CS setup
        self.chip = gpiod.chip("0")
        self.cs_line = self.chip.get_line(self.cs_pin)
        cs_line_out = gpiod.line_request()
        cs_line_out.consumer = "SPI_CS"
        cs_line_out.request_type = gpiod.line_request.DIRECTION_OUTPUT
        self.cs_line.request(cs_line_out)
        self.cs_line.set_value(1)

        # SPI chip 0 (channels 1-8)
        self.spi = spidev.SpiDev()
        self.spi.open(0, 0)
        self.spi.max_speed_hz = 4000000
        self.spi.lsbfirst = False
        self.spi.mode = 0b01
        self.spi.bits_per_word = 8

        # SPI chip 1 (channels 9-16)
        self.spi_2 = spidev.SpiDev()
        self.spi_2.open(0, 1)
        self.spi_2.max_speed_hz = 4000000
        self.spi_2.lsbfirst = False
        self.spi_2.mode = 0b01
        self.spi_2.bits_per_word = 8

        self.result   = [0] * 27
        self.result_2 = [0] * 27

        self._init_chip_1()
        self._init_chip_2()

    # Let's make this a context manager
    # that way we arent wasting resources by mistake

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_val, exception_traceback):
        self.close()

    # --- Chip 1 helpers ---
    def _send_command(self, command):
        self.spi.xfer([command])

    def _write_byte(self, register, data):
        register_write = 0x40 | register
        self.spi.xfer([register_write, 0x00, data])

    # --- Chip 2 helpers ---
    # Note here that the second ADS1992 chip is
    # not on the usual RPi SPI chip select so
    # it is not handled by the hardware and is
    # handled manually by this script
    def _send_command_2(self, command):
        self.cs_line.set_value(0)
        self.spi_2.xfer([command])
        self.cs_line.set_value(1)

    def _write_byte_2(self, register, data):
        register_write = 0x40 | register
        self.cs_line.set_value(0)
        self.spi_2.xfer([register_write, 0x00, data])
        self.cs_line.set_value(1)

    def _init_chip_1(self):
        self._send_command(self.WAKEUP)
        self._send_command(self.STOP)
        self._send_command(self.RESET)
        self._send_command(self.SDATAC)

        self._write_byte(0x14, 0x80)
        self._write_byte(self.CONFIG1, 0x96)
        self._write_byte(self.CONFIG2, 0xD4)
        self._write_byte(self.CONFIG3, 0xFF)
        self._write_byte(0x04, 0x00)
        self._write_byte(0x0D, 0x00)
        self._write_byte(0x0E, 0x00)
        self._write_byte(0x0F, 0x00)
        self._write_byte(0x10, 0x00)
        self._write_byte(0x11, 0x00)
        self._write_byte(0x15, 0x20)
        self._write_byte(0x17, 0x00)
        self._write_byte(self.CH1SET, 0x00)
        self._write_byte(self.CH2SET, 0x00)
        self._write_byte(self.CH3SET, 0x00)
        self._write_byte(self.CH4SET, 0x00)
        self._write_byte(self.CH5SET, 0x00)
        self._write_byte(self.CH6SET, 0x00)
        self._write_byte(self.CH7SET, 0x00)
        self._write_byte(self.CH8SET, 0x00)

        self._send_command(self.RDATAC)
        self._send_command(self.START)

    def _init_chip_2(self):
        self._send_command_2(self.WAKEUP)
        self._send_command_2(self.STOP)
        self._send_command_2(self.RESET)
        self._send_command_2(self.SDATAC)

        self._write_byte_2(0x14, 0x80)
        self._write_byte_2(self.CONFIG1, 0x96)
        self._write_byte_2(self.CONFIG2, 0xD4)
        self._write_byte_2(self.CONFIG3, 0xFF)
        self._write_byte_2(0x04, 0x00)
        self._write_byte_2(0x0D, 0x00)
        self._write_byte_2(0x0E, 0x00)
        self._write_byte_2(0x0F, 0x00)
        self._write_byte_2(0x10, 0x00)
        self._write_byte_2(0x11, 0x00)
        self._write_byte_2(0x15, 0x20)
        self._write_byte_2(0x17, 0x00)
        self._write_byte_2(self.CH1SET, 0x00)
        self._write_byte_2(self.CH2SET, 0x00)
        self._write_byte_2(self.CH3SET, 0x00)
        self._write_byte_2(self.CH4SET, 0x00)
        self._write_byte_2(self.CH5SET, 0x00)
        self._write_byte_2(self.CH6SET, 0x00)
        self._write_byte_2(self.CH7SET, 0x00)
        self._write_byte_2(self.CH8SET, 0x00)

        self._send_command_2(self.RDATAC)
        self._send_command_2(self.START)

    def _parse_frame(self, raw, result_buf):
        for a in range(3, 25, 3):
            voltage = (raw[a] << 8) | raw[a + 1]
            voltage = (voltage << 8) | raw[a + 2]
            convert = voltage | self.DATA_TEST
            if convert == self.DATA_CHECK:
                voltage_converted = voltage - 16777214
            else:
                voltage_converted = voltage
            channel_num = int(a / 3)
            result_buf[channel_num] = round(1000000 * 4.5 * (voltage_converted / 16777215), 2)

    def read_sample(self):
        """
        Reads one frame from both chips.
        Returns a list of 16 µV floats (channels 1-16),
        or None if the frame header is invalid.
        """
        output = self.spi.readbytes(27)

        if (output[0] == 192 and output[1] == 0 and output[2] == 8):
            self._parse_frame(output, self.result);

        self.cs_line.set_value(0)
        output_2 = self.spi_2.readbytes(27)
        self.cs_line.set_value(1)

        if (output_2[0] == 192 and output_2[1] == 0 and output_2[2] == 8):
            self._parse_frame(output_2, self.result_2)

            return self.result[1:9] + self.result_2[1:9]


    def close(self):
        self.spi.close()
        self.spi_2.close()


# # Sample driver code
# # uncomment before using
# print("Initialising EEGReader...")
# try:
#     eeg = EEGReader()
#     print("✓ Init complete")
# except Exception as e:
#     print(f"✗ Init failed: {e}")
#     exit(1)

# print("Reading 10000 samples...")
# try:
#     for i in range(10000):
#         sample = eeg.read_sample()
#         if sample is None:
#             print(f"  [{i}] Invalid frame header — skipping")
#         else:
#             print(f"  [{i}] {sample}")
# except Exception as e:
#     print(f"✗ Read failed: {e}")
# finally:
#     eeg.close()
#     print("SPI closed")
