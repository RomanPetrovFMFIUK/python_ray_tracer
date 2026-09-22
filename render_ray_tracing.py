from numba import njit, prange
from objects_for_ray_tracing import Scene
from settings import WIDTH, HEIGHT
from raytracing_math import (scale_to_canvas,
                             trace_ray,
                             convert_spheres_to_matrics,
                             convert_lights_to_matrix)
import numpy as np
import pygame
import time

@njit(parallel=True, fastmath=True)
def compute_image(spheres, lights, camera_pos, r_matrix):

    o = camera_pos

    img = np.zeros((WIDTH, HEIGHT, 3), dtype=np.uint8)

    for idx in prange(WIDTH * HEIGHT):
        x = idx // HEIGHT - WIDTH // 2
        y = idx % HEIGHT - HEIGHT // 2

        d = scale_to_canvas(x, y)
        d = np.dot(r_matrix, d)

        color = trace_ray(o, d, spheres, lights, 1.0, float('inf'), 10)

        screen_x = int(WIDTH / 2 + x)

        screen_y = int(HEIGHT / 2 - y - 1)

        if 0 <= screen_x < WIDTH and 0 <= screen_y < HEIGHT:
            r = min(max(int(color[0]), 0), 255)
            g = min(max(int(color[1]), 0), 255)

            b = min(max(int(color[2]), 0), 255)

            img[screen_x, screen_y, 0] = r
            img[screen_x, screen_y, 1] = g
            img[screen_x, screen_y, 2] = b
        # print(x)


    return img


def render(screen: pygame.Surface, scene: Scene, camera_pos, r_matrix):
    print('Start Render')

    spheres = convert_spheres_to_matrics(scene.objects)
    lights = convert_lights_to_matrix(scene.lights)

    start_time = time.perf_counter()
    img_array = compute_image(spheres, lights, camera_pos, r_matrix)
    pygame.surfarray.blit_array(screen, img_array)
    time_end = time.perf_counter()
    print("---Time: %s seconds ---" % (time_end - start_time))
    print('Render complete')
