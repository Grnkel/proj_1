from image import ImageHandler
from terminal import TerminalHandler
from functools import partial
from ascii import Ascii
import time

def main1():
    image = ImageHandler('images/image1.jpg')
    ascii = Ascii('chars/font4x6.png')
    ascii.generate_list()
    
    height, width = ascii.chunk_dims
    image.fit_chunk(height, width)

    timer = time.perf_counter_ns()
    image.apply(cores=1,func=partial(ascii.ascii_print, image.image))
    print("time taken:", (time.perf_counter_ns() - timer) * 10**-6, "ms")
    image.show()

def main2():
    terminal = TerminalHandler('images/image1.jpg')

    timer = time.perf_counter_ns()
    GRAY = True
    GRAY = False
    if GRAY:
        terminal.grayscale().apply(func=partial(terminal.contrast,15,0.5))
    else:
        terminal.apply(func=partial(terminal.contrast,12,0.5))
        pass

    terminal.to_terminal()
    print("time taken:", (time.perf_counter_ns() - timer) * 10**-6, "ms")

main1()

# TODO kolla hur mycket space (v och h som finns och ta mindre och mindre chars
# TODO skapa funktionalitet för videos

# TODO gör den snabbare och mer effektiv (bättre lösning) 
# alltså kanske att ascii skiten är en lista med bools eller 
# något där man faktiskt får veta att det antingen är 0 eller 1 
# ör att verkligen snabba upp beräkningen, kanske först därefter 
# som man kollar färg? testa att sänka precisionen på alla matriser, 
# man behöver ju inte tre kanaler med sådan där stor precision direkt, 
# man kanske kan göra bit-manipulationer eller vem vet vad python kan göra

# TODO ta in streams från din webcam
# TODO varför är det 256 överallt!?
# TODO ta bort dims och överflödiga saker
# TODO något är verkligen fel med hur processorn fixar multicore




