import time
from rpi_ws281x import PixelStrip, Color

LED_COUNT = 7
LED_PIN = 12
LED_FREQ_HZ = 800000
LED_DMA = 10
LED_BRIGHTNESS = 120
LED_INVERT = False
LED_CHANNEL = 0

strip = PixelStrip(
    LED_COUNT,
    LED_PIN,
    LED_FREQ_HZ,
    LED_DMA,
    LED_INVERT,
    LED_BRIGHTNESS,
    LED_CHANNEL
)
strip.begin()

for i in range(LED_COUNT):
    strip.setPixelColor(i, Color(255, 255, 255))  # White

strip.show()

while True:
    time.sleep(1)
