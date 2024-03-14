from pyflyer import Aircraft, World
import numpy as np
from PIL import Image


def test_aircraft():
    ac = Aircraft()
    ac.reset([0.0, 0.0, 1000.0], 0.0, 100.0)
    dt = 0.001
    input_state = [0.0, 0.0, 0.0, 0.0]
    ac.step(dt, input_state)

    pos = ac.position
    vel = ac.velocity
    att = ac.attitude
    rate = ac.rates
    print(f"pos: {pos}, vel: {vel}, att: {att}, rate: {rate}")


def test_world():
    world = World()
    ac = Aircraft()
    world.add_aircraft(ac)
    world.create_map(5, [128, 128], 25.0, True)

    world.render()
    arr = np.array(world.get_image(), dtype=np.uint8).reshape((1024, 1024, 4))
    swap_arr = np.copy(arr)
    swap_arr[:, :, 0], swap_arr[:, :, 2] = arr[:, :, 2], arr[:, :, 0]
    image = Image.fromarray(swap_arr[:, :, 0:3], mode="RGB")
    image.save("world.jpeg")


if __name__ == "__main__":
    test_aircraft()
